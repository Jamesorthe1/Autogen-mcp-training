import google.generativeai as genai
import os
from PIL import Image
import torch
import numpy as np
import faiss
import spacy
from transformers import AutoTokenizer, AutoProcessor
import requests
import re
import traceback
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from googleapiclient.discovery import build
from io import BytesIO

# Import functions from tools.py
from tools import display_map_for_location, display_travel_time_and_route, search_images, display_images


# Load environment variables (ensure this is done in each file that needs them)
load_dotenv()

# Load a spaCy model (download if not already present)
try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model 'en_core_web_sm' downloaded and loaded.")

# Hardcoded values for fallback (should ideally be in a config or env)
# These are now primarily used within tools.py functions, but kept here for completeness
HARDCODED_GOOGLE_CLOUD_API_KEY = os.getenv('GOOGLE_CLOUD_API_KEY', 'AIzaSyAECC4ASkB6uOZMQXkbiNYsaHRGf1_K6k') # Use env var as first priority
HARDCODED_PROGRAMMABLE_SEARCH = os.getenv('PROGRAMMABLE_SEARCH', '848db169675dd476e') # Use env var as first priority


class SimpleAIAgent:
    """
    An AI agent capable of understanding user intent, deciding on actions,
    and orchestrating a multimodal RAG workflow. It uses a multimodal embedding
    model, FAISS indexes for retrieval, and integrates with external tools
    (maps, travel time, image search) and a large language model (Gemini)
    for response generation.
    """
    def __init__(self, name, embedding_model, tokenizer, processor, modality_indexes, modality_data_items, text_feature_dim, image_feature_dim, audio_feature_dim):
        """
         Initializes the SimpleAIAgent with necessary components for RAG and tool use.

        Args:
            name (str): The name of the agent.
            embedding_model: An instance of the MultimodalEmbeddingModel class.
            tokenizer: The tokenizer for the embedding model (e.g., CLIP tokenizer).
            processor: The processor for the embedding model (e.g., CLIP processor).
            modality_indexes (dict): A dictionary where keys are modality names (e.g., 'text', 'image', 'audio_segment')
                                     and values are FAISS indexes for that modality's embeddings.
            modality_data_items (dict): A dictionary where keys are modality names and values are lists
                                        of the original data items corresponding to the embeddings in the index.
            text_feature_dim (int): Dimension of text embeddings.
            image_feature_dim (int): Dimension of image embeddings.
            audio_feature_dim (int): Dimension of audio embeddings.
        """
        self.name = name
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.processor = processor
        self.modality_indexes = modality_indexes
        self.modality_data_items = modality_data_items
        self.text_feature_dim = text_feature_dim
        self.image_feature_dim = image_feature_dim
        self.audio_feature_dim = audio_feature_dim

        # Check Gemini API configuration once during initialization
        self.gemini_configured = False
        try:
            api_key_name = 'NEW_GOOGLE_API_KEY' # Use the new secret name
            # In a local script, we get from environment variables
            google_api_key = os.getenv(api_key_name)

            if google_api_key:
                genai.configure(api_key=google_api_key)
                # Check if any models supporting generateContent are available
                # Filter for models that are actually Model objects and support generateContent
                available_models = [m.name for m in genai.list_models() if isinstance(m, genai.types.Model) and 'generateContent' in m.supported_generation_methods]

                if available_models:
                     self.gemini_configured = True
                     print(f"Gemini API configured successfully for agent '{self.name}'. Available models supporting generateContent: {available_models}")
                else:
                     print(f"Gemini API key found and configured for agent '{self.name}', but no models supporting generateContent were listed. Response generation may fail.")
            else:
                print(f"Gemini API key '{api_key_name}' not found in environment variables for agent '{self.name}'. Response generation will use a placeholder.")
        except Exception as e:
            print(f"Error during Gemini API configuration for agent '{self.name}': {e}. Response generation will use a placeholder.")


    def understand_intent(self, query: str):
        """
        Analyzes the user query to determine the user's intent using spaCy for NER
        and keyword matching.

        Args:
            query (str): The user's query string.

        Returns:
            str: A string representing the identified intent (e.g., 'perform_rag',
                 'display_map', 'calculate_travel_time', 'search_images', 'respond_general').
        """
        # Process query for location entities first using spaCy
        doc = nlp(query.lower())
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        print(f"Debug: spaCy found {len(locations)} locations: {locations}") # Debug print

        # Prioritize travel time/route intent if two locations are clearly indicated
        travel_keywords = ["travel time", "how long to get to", "route from", "directions from", "distance from", "get from", "to"]
        # Check if any travel keywords are present OR if at least two locations are found by spaCy
        if any(keyword in query.lower() for keyword in travel_keywords) or len(locations) >= 2:
             print("Debug: Identified potential travel time/route intent keywords or found two locations.") # Debug print
             source, destination = self.extract_source_destination_from_query(query)
             if source and destination:
                  print("Debug: Successfully extracted source and destination for travel time intent.") # Debug print
                  return "calculate_travel_time"
             else:
                  # If keywords are present but two locations aren't clearly extracted,
                  # it might still be a RAG query about travel.
                  print("Debug: Could not extract two locations for travel time intent. Falling back to next intent check.") # Debug print


        # Check for image search intent keywords
        image_keywords = ["show me images of", "images of", "pictures of", "show me pictures of", "find images of", "find pictures of"]
        if any(keyword in query.lower() for keyword in image_keywords):
             print("Debug: Identified image search intent keywords.") # Debug print
             return "search_images"

        # Check for map display intent keywords or if a single location is mentioned without other specific intents
        map_keywords = ["map", "show map of", "where is"]
        # Check if any map keywords are present OR if exactly one location is found by spaCy AND no travel/image keywords are present
        if any(keyword in query.lower() for keyword in map_keywords) or (len(locations) == 1 and not any(keyword in query.lower() for keyword in travel_keywords + image_keywords)):
            print("Debug: Identified map intent keywords or found a single location without other specific intents.") # Debug print
            location = self.extract_location_from_query(query)
            if location:
                print(f"Debug: Extracted single location '{location}', confirming map display intent.")
                return "display_map"
            else:
                # If map keywords are present but no single location is extracted,
                # it might be a general query about maps or locations for RAG.
                print("Debug: Map keyword found but no single location extracted. Falling back to RAG.")


        # Default to RAG if no specific tool intent is detected
        print("Debug: No specific intent keywords or clear location-based intent detected. Defaulting to RAG.") # Debug print
        return "perform_rag"


    def decide_action(self, intent: str):
        """
        Decides the appropriate action based on the identified intent.
        Currently, this is a simple mapping. More complex logic could involve
        checking available tools, data, user preferences, etc.

        Args:
            intent (str): The identified intent string from `understand_intent`.

        Returns:
            str: A string representing the chosen action ('perform_rag',
                 'display_map', 'calculate_travel_time', 'search_images', 'respond_general').
        """
        # Simple mapping of intent to action for now
        if intent in ["display_map", "calculate_travel_time", "search_images"]:
            return intent
        else:
            return "perform_rag" # Default to RAG for all other intents


    def extract_location_from_query(self, query: str):
        """
        Extracts a single location name from the user query using spaCy and regex patterns.

        Args:
            query (str): The user's query string.

        Returns:
            str or None: A string representing the extracted location name, or None if not found.
        """
        doc = nlp(query)
        # Look for named entities that are locations (GPE, LOC)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]

        # Prioritize locations identified by spaCy
        if locations:
            print(f"Debug: spaCy extracted location(s) in extract_location_from_query: {locations}") # Debug print
            return locations[0] # Return the first location found


        # Fallback to keyword-based extraction for common patterns if spaCy didn't find any
        print("Debug: No location extracted by spaCy in extract_location_from_query. Attempting regex.")
        # Regex to capture text after "map of", "where is", "show map of"
        match = re.search(r'(?:map of|where is|show map of)\s+(.+)', query, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            print(f"Debug: Regex matched map pattern in extract_location_from_query, extracted: {extracted}") # Debug print
            return extracted
        # Add a pattern to capture just a single location name if no other keywords are present
        # This regex captures text from the start to end if no commas are present,
        # assuming a single location name is provided directly.
        match = re.search(r'^\s*([^,]+?)\s*$', query)
        if match:
            extracted = match.group(1).strip()
            # Avoid extracting very short or generic terms that are unlikely locations
            if len(extracted) > 2 and extracted.lower() not in ["the", "a", "an", "what", "where", "is", "of", "show", "me", "images", "pictures", "route", "time", "distance", "from", "to", "get"]:
                 print(f"Debug: Regex matched single location pattern in extract_location_from_query, extracted: {extracted}") # Debug print
                 return extracted
        print("Debug: No location extracted by spaCy or regex in extract_location_from_query.") # Debug print
        return None


    def extract_source_destination_from_query(self, query: str):
        """
        Extracts source and destination locations from a travel time/route query
        using spaCy and regex patterns.

        Args:
            query (str): The user's query string.

        Returns:
            tuple: A tuple containing (source_location, destination_location),
                   or (None, None) if not found.
        """
        doc = nlp(query)
        # Look for named entities that are locations (GPE, LOC)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        print(f"Debug: spaCy found {len(locations)} locations in extract_source_destination_from_query: {locations}") # Debug print

        # Prioritize regex patterns that explicitly mention source and destination (e.g., "from X to Y")
        match = re.search(r'from\s+(.+)\s+to\s+(.+)', query, re.IGNORECASE)
        if match:
            source = match.group(1).strip()
            destination = match.group(2).strip()
            print(f"Debug: Regex matched 'from...to...' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}'") # Debug print
            return source, destination
        match = re.search(r'route from\s+(.+)\s+to\s+(.+)', query, re.IGNORECASE)
        if match:
             source = match.group(1).strip()
             destination = match.group(2).strip()
             print(f"Debug: Regex matched 'route from...to...' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}'") # Debug print
             return source, destination
        match = re.search(r'travel time from\s+(.+)\s+to\s+(.+)', query, re.IGNORECASE)
        if match:
             source = match.group(1).strip()
             destination = match.group(2).strip()
             print(f"Debug: Regex matched 'travel time from...to...' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}'") # Debug print
             return source, destination
        match = re.search(r'directions from\s+(.+)\s+to\s+(.+)', query, re.IGNORECASE) # Added directions pattern
        if match:
             source = match.group(1).strip()
             destination = match.group(2).strip()
             print(f"Debug: Regex matched 'directions from...to...' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}')") # Debug print
             return source, destination
        match = re.search(r'distance from\s+(.+)\s+to\s+(.+)', query, re.IGNORECASE) # Added distance pattern
        if match:
             source = match.group(1).strip()
             destination = match.group(2).strip()
             print(f"Debug: Regex matched 'distance from...to...' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}')") # Debug print
             return source, destination
        # Added a pattern for "location x to location y" without explicit 'from' or 'to' keywords, relying more on spaCy's entity recognition
        match = re.search(r'([^,]+?)\s+to\s+([^,]+)', query, re.IGNORECASE)
        if match:
            source_potential = match.group(1).strip()
            destination_potential = match.group(2).strip()
            # Validate if these extracted parts look like locations using spaCy entities
            doc_potential = nlp(f"{source_potential} to {destination_potential}")
            potential_locations = [ent.text for ent in doc_potential.ents if ent.label_ in ["GPE", "LOC"]]
            if len(potential_locations) >= 2:
                 source = potential_locations[0]
                 destination = potential_locations[1]
                 print(f"Debug: Regex and spaCy matched 'location x to location y' pattern in extract_source_destination_from_query: Source='{source}', Destination='{destination}')") # Debug print
                 return source, destination
            else:
                 print(f"Debug: Regex matched 'location x to location y' pattern but spaCy did not confirm two locations in extract_source_destination_from_query.")


        # Fallback: If no explicit patterns found, try to use the first two location entities found by spaCy
        if len(locations) >= 2:
            print(f"Debug: Using first two spaCy locations as fallback in extract_source_destination_from_query: Source='{locations[0]}', Destination='{locations[1]}'") # Debug print
            return locations[0], locations[1]


        print("Debug: No source and destination extracted by spaCy or regex patterns in extract_source_destination_from_query.") # Debug print
        return None, None # Return None, None if extraction fails


    def generate_query_embedding(self, query: str):
        """
        Generates an embedding for the user query using the text embedding model.

        Args:
            query (str): The user's query string.

        Returns:
            np.ndarray: A NumPy array representing the query embedding, or a zero vector
                        if embedding generation fails or returns a non-1D shape.
        """
        if self.embedding_model is None or self.tokenizer is None:
            print("Error: Embedding model or tokenizer not initialized. Cannot generate query embedding.")
            return np.zeros(self.text_feature_dim, dtype='float32') # Return zero vector if dependencies are missing

        try:
            # Use the tokenizer and embedding model to get the text embedding
            # Assume the embedding_model has a get_embedding method that handles text
            query_embedding = self.embedding_model.get_embedding(query, 'text')

            if query_embedding is not None:
                 # Ensure the output is a 1D NumPy array
                 if isinstance(query_embedding, torch.Tensor):
                      query_embedding = query_embedding.squeeze().cpu().numpy()
                 elif isinstance(query_embedding, np.ndarray):
                      query_embedding = query_embedding.squeeze()

                 # Validate the shape of the final embedding
                 if query_embedding.ndim == 1 and query_embedding.shape[0] == self.text_feature_dim:
                      return query_embedding
                 else:
                      print(f"Warning: Generated query embedding has unexpected shape ({query_embedding.shape}) after processing. Expected 1D with dimension {self.text_feature_dim}. Returning zero vector.")
                      return np.zeros(self.text_feature_dim, dtype='float32') # Return zero vector if shape is incorrect


            else:
                 print("Warning: Embedding generation returned None. Returning zero vector.")
                 return np.zeros(self.text_feature_dim, dtype='float32') # Return zero vector if embedding is None


        except Exception as e:
            print(f"Error generating query embedding: {e}")
            traceback.print_exc()
            # Return a zero vector or handle appropriately if embedding fails
            return np.zeros(self.text_feature_dim, dtype='float32') # Return a zero vector of the expected dimension




    def retrieve_relevant_data(self, query_embedding, query_text: str, k_per_modality: int = 2, relevance_threshold: float = 50.0):
        """
        Retrieves relevant data by searching modality-specific FAISS indexes and
        includes a web search fallback if no relevant local data is found.

        Args:
            query_embedding (np.ndarray): The embedding of the user's query (1D NumPy array).
            query_text (str): The original text of the user's query.
            k_per_modality (int): The number of nearest neighbors to retrieve from each local index.
            relevance_threshold (float): A threshold for considering a local result relevant (lower distance is more relevant).

        Returns:
            list: A list of relevant data items, potentially including items from the local indexes
                  and web search results. Each item will be a dictionary with a 'source' key
                  ('local_index' or 'web_search') and a 'data' key containing the retrieved information.
                  Local index items also include 'modality' and 'distance'.
        """
        relevant_items = []

        # 1. Query the relevant FAISS index(es).
        if query_embedding is not None and self.modality_indexes:
            print("Agent searching modality-specific FAISS indexes...")
            # Ensure query_embedding is float32 and 2D for FAISS search
            query_embedding_reshaped = np.array(query_embedding).reshape(1, -1).astype('float32')

            for modality_name, index in self.modality_indexes.items():
                if index and index.ntotal > 0:
                    print(f"Searching {modality_name} index (dimension {index.d})...")
                    try:
                        # Check if query embedding dimension matches the index dimension.
                        if query_embedding_reshaped.shape[-1] != index.d:
                             print(f"Warning: Query embedding dimension ({query_embedding_reshaped.shape[-1]}) does not match actual {modality_name} index dimension ({index.d}). Skipping search for this modality.")
                             continue

                        # Perform FAISS search
                        distances, indices = index.search(query_embedding_reshaped, k_per_modality)

                        print(f"FAISS search found {len(indices[0])} potential neighbors in {modality_name} index.")

                        # 4. Store the search results for each queried modality.
                        # 5. Combine the results from all queried modalities.
                        for i in range(len(indices[0])):
                            idx = indices[0][i]
                            distance = distances[0][i]

                            # Check if the index is valid and within bounds of the data items list for this modality
                            if idx != -1 and 0 <= idx < len(self.modality_data_items.get(modality_name, [])):
                                 # Check if the distance is below the relevance threshold
                                 if distance < relevance_threshold:
                                    # Retrieve the actual data item using the index
                                    data_item = self.modality_data_items[modality_name][idx]
                                    relevant_items.append({'source': 'local_index', 'modality': modality_name, 'data': data_item, 'distance': distance})
                                    print(f"Found relevant item in {modality_name} index: {data_item.get('id', 'N/A')} with distance {distance:.4f}")
                                 else:
                                    print(f"Item {self.modality_data_items[modality_name][idx].get('id', 'N/A')} found with distance {distance:.4f} in {modality_name} index, above relevance threshold ({relevance_threshold:.4f}).")
                            elif idx != -1:
                                # This case should ideally not happen if the bounds check is correct,
                                # but kept for robustness.
                                print(f"Item from {modality_name} index found with distance {distance:.4f}, above relevance threshold {relevance_threshold:.4f}.")
                            else:
                                print(f"Invalid index (-1) found in {modality_name} index search results.")


                    except Exception as e:
                        print(f"Error searching {modality_name} index: {e}")
                        traceback.print_exc()
                else:
                    print(f"No index or empty index for modality: {modality_name}. Skipping search.")

            # 6. Sort the combined list of local results by distance (lower distance is more relevant).
            relevant_items.sort(key=lambda x: x.get('distance', float('inf')))
            print(f"Combined and sorted {len(relevant_items)} potential local results.")


        # --- Web Search Fallback ---
        # Perform web search if no relevant local items are found after searching all applicable indexes
        if not relevant_items:
            print("No relevant local data found. Performing web search...")
            try:
                # Use environment variables for API keys, fallback to hardcoded
                google_api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
                programmable_search_engine_id = os.getenv('PROGRAMMABLE_SEARCH')

                # Fallback to hardcoded values if environment variables are not set
                if not google_api_key:
                     google_api_key = HARDCODED_GOOGLE_CLOUD_API_KEY
                     print("Using hardcoded GOOGLE_CLOUD_API_KEY for web search.")
                if not programmable_search_engine_id:
                     programmable_search_engine_id = HARDCODED_PROGRAMMABLE_SEARCH
                     print("Using hardcoded PROGRAMMABLE_SEARCH ID for web search.")


                if not google_api_key or not programmable_search_engine_id:
                    print("Google Search API key or Programmable Search Engine ID not available (neither from env vars nor hardcoded). Cannot perform web search.")
                    return relevant_items # Return empty list if keys are missing

                service = build("customsearch", "v1", developerKey=google_api_key)
                # Perform a general web search for the query text
                res = service.cse().list(q=query_text, cx=programmable_search_engine_id, num=k_per_modality).execute()

                if 'items' in res:
                    for item in res['items']:
                        # Structure web search results similarly for consistent processing
                        relevant_items.append({'source': 'web_search', 'data': item})
                    print(f"Added {len(res['items'])} web search results.")
                else:
                    print("Web search returned no items.")

            except Exception as e:
                print(f"Error during web search: {e}")
                traceback.print_exc()

        return relevant_items


    def generate_response(self, query: str, relevant_items: list):
        """
        Generates a response using a multimodal LLM (Gemini 1.5 Flash) based on the query and retrieved context.

        Args:
            query (str): The original user query string.
            relevant_items (list): A list of dictionaries containing retrieved relevant data
                                   (from local index or web search).

        Returns:
            str: A string representing the generated response from the LLM, or an error message
                 if the Gemini API is not configured or encounters an error.
        """
        # Prepare content for the Gemini model, including multimodal parts
        # The prompt guides the LLM to use the provided context.
        gemini_content = [
            f"You are an AI assistant specializing in tourism and travel information. Your goal is to answer the user's query concisely and helpfully, drawing only from the provided context. If the context is insufficient, state that you cannot fully answer based on the available information. Combine information from different sources in the context to form a comprehensive answer.\n\n",
            f"User Query: {query}\n\n",
            "Retrieved Context:\n"
        ]

        if relevant_items:
            for i, item in enumerate(relevant_items):
                gemini_content.append(f"--- Context Item {i+1} (Source: {item.get('source', 'Unknown')}) ---\n")
                data_item = item.get('data', {})

                if item.get('source') == 'local_index' and isinstance(data_item, dict):
                    # Include details from local index items
                    gemini_content.append(f"Data ID: {data_item.get('id', 'N/A')}\n")
                    gemini_content.append(f"Modality: {item.get('modality', 'Unknown')}\n") # Include modality in context
                    if 'text' in data_item and data_item['text']:
                        # Append text content, truncating if long
                        gemini_content.append(f"Text Content: {data_item['text'][:1000]}...\n") # Truncate text for context (increased length)
                    if 'image' in data_item and isinstance(data_item['image'], Image.Image):
                        # Append the actual image object for Gemini to analyze
                        gemini_content.append(data_item['image'])
                        gemini_content.append("\n") # Add a newline after the image
                        print(f"Agent included image '{data_item.get('id', 'N/A')}' in Gemini prompt.")
                    # Handle video frames which are stored as images
                    if item.get('modality') == 'video_frame' and 'image' in data_item and isinstance(data_item['image'], Image.Image):
                         # Append the actual video frame image object
                         gemini_content.append(data_item['image'])
                         gemini_content.append("\n") # Add a newline after the video frame
                         print(f"Agent included video frame '{data_item.get('id', 'N/A')}' in Gemini prompt.")
                    # Note: Gemini's current multimodal capabilities primarily focus on text and images.
                    # Audio data cannot be directly included in the same way. You would need
                    # a separate audio-specific model or a multimodal model that supports audio.
                    # For now, we'll just note its presence in the text context if audio_path is present.
                    if item.get('modality') == 'audio_segment' and 'audio_path' in data_item:
                         gemini_content.append("Audio Data: Present (Note: Audio data is not directly processed by this multimodal model)\n")
                         print(f"Agent noted presence of audio data '{data_item.get('id', 'N/A')}' in text context.")

                    if 'distance' in item:
                         gemini_content.append(f"Retrieval Distance: {item['distance']:.4f}\n")

                elif item.get('source') == 'web_search' and isinstance(data_item, dict):
                     # Include details from web search results
                     if 'title' in data_item: gemini_content.append(f"Web Search Title: {data_item['title']}\n")
                     if 'body' in data_item: gemini_content.append(f"Web Search Snippet: {data_item['body'][:500]}...\n") # Truncate snippet
                     if 'href' in data_item: gemini_content.append(f"Web Search URL: {data_item['href']}\n")

                else:
                    # Handle unexpected item formats
                    gemini_content.append(f"Agent received unexpected data format for item: {item}\n")

                gemini_content.append("---\n") # Separator between context items
        else:
            gemini_content.append("No relevant context found.")

        gemini_content.append(f"\nGenerated Response:") # Prompt for the LLM response


        # --- Integrate Gemini API for Response Generation ---
        # Ensure the Gemini API is configured before attempting to use it.
        if not self.gemini_configured:
            # Return a text-only response including the context if API is not configured
            text_only_response = "".join([part if isinstance(part, str) else "" for part in gemini_content])
            return f"Gemini API key not configured for agent '{self.name}'. Cannot generate LLM response.\n\n{text_only_response}"


        try:
            # Initialize the Gemini model (use the specific 1.5 Flash model name)
            # Check the output of genai.list_models() for available names.
            # Using a model name that supports generateContent.
            model_name = 'gemini-1.5-flash-latest' # Or other available 1.5 Flash name
            model = None # Initialize model to None before finding and initializing

            # Check if the model is available and supports generateContent
            # Add a check to ensure 'm' is a model object before accessing '.name'
            available_models = [m.name for m in genai.list_models() if isinstance(m, genai.types.Model) and 'generateContent' in m.supported_generation_methods]
            if model_name not in available_models:
                 print(f"Warning: Requested model '{model_name}' not available or does not support generateContent. Trying another flash model if available.")
                 # Try to find another flash model
                 flash_models = [name for name in available_models if 'flash' in name.lower()]
                 if flash_models:
                     alternative_model_name = flash_models[0] # Get the name of the first available flash model
                     try:
                         model = genai.GenerativeModel(alternative_model_name)
                         print(f"Using available flash model: '{alternative_model_name}' for text generation.")
                     except Exception as e:
                         print(f"Error initializing alternative flash model '{alternative_model_name}': {e}")
                 else:
                     print("No suitable Gemini Flash model found for text generation.")


            if model: # Check if model was successfully initialized
                 # Construct the prompt for the Gemini model
                 prompt_parts = gemini_content # Use the prepared list directly


                 # Generate content using the Gemini model
                 # Handle potential errors during generation
                 try:
                     gemini_response = model.generate_content(prompt_parts)
                     return gemini_response.text
                 except Exception as e:
                     print(f"Error during model.generate_content: {e}")
                     # Define context_text here in case of an error before it's fully built
                     context_text = "".join([part if isinstance(part, str) else "" for part in gemini_content])
                     return f"Error generating response from Gemini API: {e}\n\n{context_text}"

            else:
                # Return placeholder if no model could be initialized
                # Define context_text here in case of an error before it's fully built
                context_text = "".join([part if isinstance(part, str) else "" for part in gemini_content])
                return f"Cannot generate response because no suitable Gemini Flash model was found or initialized.\n\n{context_text}"


        except Exception as e:
            # Define context_text here in case of an error before it's fully built
            context_text = "".join([part if isinstance(part, str) else "" for part in gemini_content])
            return f"Error calling Gemini API during response generation setup: {e}\n\n{context_text}"


    def orchestrate_workflow(self, query: str):
        """
        Orchestrates the workflow based on the user's query.
        This is the main control flow for the agent, determining intent,
        deciding action, executing the action (RAG or tool use), and
        generating a response.

        Args:
            query (str): The user's query string.
        """
        print(f"\nAgent '{self.name}' received query: '{query}'")

        # 1. Understand Intent
        intent = self.understand_intent(query)
        print(f"Agent identified intent: '{intent}'")

        # 2. Decide Action
        action = self.decide_action(intent)
        print(f"Agent decided action: '{action}'")

        # 3. Execute Action based on the decided action
        if action == "perform_rag":
            print("Agent performing Multimodal RAG...")
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)

            # Retrieve relevant data using modality-specific indexes and data items
            # RELEVANCE_THRESHOLD is defined in main.py, need to pass it or define here
            # For now, let's use a default or pass it
            RELEVANCE_THRESHOLD = 40.0 # Define a default or pass from main
            relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)

            # Generate response using LLM based on retrieved data
            response = self.generate_response(query, relevant_items)
            print(f"\nAgent Response:\n{response}")

        elif action == "display_map":
            print("Agent executing 'display_map' tool...")
            # Extract location from the query
            location = self.extract_location_from_query(query)
            if location:
                 # Call the external function from tools.py to display the map
                 display_map_for_location(location)
                 # Optionally, perform a related image search and display images
                 image_query = f"images of {location}"
                 print(f"Agent considering related image search for: '{image_query}'")
                 # Call search_images and then display_images from tools.py
                 image_urls = search_images(image_query)
                 if image_urls:
                      print(f"Found {len(image_urls)} images for '{image_query}'. Displaying...")
                      display_images(image_urls)
                 else:
                     print(f"No images found for '{image_query}'.")


            else:
                print("Agent could not extract a location from the query for map display. Falling back to RAG.")
                # Fallback to RAG or general response if location extraction fails
                query_embedding = self.generate_query_embedding(query)
                # Pass the modality_indexes and modality_data_items to the retrieve function
                RELEVANCE_THRESHOLD = 40.0 # Define a default or pass from main
                relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)
                response = self.generate_response(query, relevant_items)
                print(f"\nAgent Response:\n{response}")


        elif action == "calculate_travel_time":
            print("Agent executing 'calculate_travel_time' tool...")
            # Extract source and destination locations from the query
            source, destination = self.extract_source_destination_from_query(query)
            if source and destination:
                # Call the external function from tools.py to display the route and time
                display_travel_time_and_route(source, destination)
            else:
                print("Agent could not extract source and destination locations for travel time calculation. Falling back to RAG.")
                # Fallback to RAG or general response if location extraction fails
                query_embedding = self.generate_query_embedding(query)
                # Pass the modality_indexes and modality_data_items to the retrieve function
                RELEVANCE_THRESHOLD = 40.0 # Define a default or pass from main
                relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)
                response = self.generate_response(query, relevant_items)
                print(f"\nAgent Response:\n{response}")


        elif action == "search_images":
             print("Agent executing 'search_images' tool...")
             # Extract the subject for image search (simple extraction for now)
             # Use regex to extract the subject more accurately after "show me images of", "images of", etc.
             match = re.search(r'(?:show me |find )?(?:images?|pictures?) of\s+(.+)', query, re.IGNORECASE)
             if match:
                 image_subject = match.group(1).strip()
                 print(f"Debug: Extracted image search subject: '{image_subject}'")
                 # Call search_images and then display_images from tools.py
                 image_urls = search_images(image_subject)
                 if image_urls:
                      print(f"Found {len(image_urls)} images for '{image_subject}'. Displaying...")
                      display_images(image_urls)
                 else:
                      print(f"No images found for '{image_subject}'.")
             else:
                 print("Agent could not identify the subject for image search. Providing a general response.")
                 # Fallback to general response if subject extraction fails
                 response = self.generate_response(query, []) # Generate response without context
                 print(f"\nAgent Response:\n{response}")


        elif action == "respond_general":
            print("Agent providing a general response.")
            # Handle general queries, potentially using the LLM without specific retrieval
            response = self.generate_response(query, []) # Generate response without context
            print(f"\nAgent Response:\n{response}")

        else: # unknown_action
            print("Agent could not determine the action for the query. Providing a general response.")
            # Handle cases where the action is not recognized
            response = self.generate_response(query, []) # Generate response without context
            print(f"\nAgent Response:\n{response}")

        print("-" * 50) # Separator