# Import necessary libraries at the beginning
import google.generativeai as genai
import os
from PIL import Image
import torch
import numpy as np
import faiss
from IPython.display import display, HTML
import spacy
from transformers import CLIPProcessor, CLIPModel
import openl3
import soundfile as sf
import folium
from geopy.geocoders import Nominatim
import requests
import polyline
import math
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pdfminer.high_level import extract_text
import cv2
from pydub import AudioSegment
from io import BytesIO # Import BytesIO for image display
import webbrowser # Import webbrowser to open HTML files
import tempfile # Import tempfile to create temporary files
import re # Import regex for more robust extraction
import traceback # Import traceback to print detailed error information
from transformers import AutoModel, AutoTokenizer, AutoProcessor # Import transformers for embedding model


load_dotenv()  # Loads .env file from project root or specified path


# --- Configure Gemini API (Local Environment) ---
# This configuration is done once at the start of the script
try:
    GOOGLE_API_KEY_ENV = os.getenv('NEW_GOOGLE_API_KEY') # Use a different variable name to avoid confusion inside functions
    if GOOGLE_API_KEY_ENV is None:
        print("Gemini API key (NEW_GOOGLE_API_KEY) not found in environment variables. Please add it to your .env file as NEW_GOOGLE_API_KEY.")
    else:
        genai.configure(api_key=GOOGLE_API_KEY_ENV)
        print("Gemini API successfully configured.")
        try:
            models = [m.name for m in genai.list_models()]
            if models:
                print("Successfully connected to Gemini API and listed models.")
            else:
                print("Warning: Connected to Gemini API, but no models were found.")
        except Exception as e:
            print(f"Error connecting to Gemini API: {e}")

except Exception as e:
    print(f"Error retrieving API key from environment or configuring Gemini API: {e}")

# --- Configure Google Search API (Local Environment) ---
# Hardcoded values for fallback
HARDCODED_GOOGLE_CLOUD_API_KEY = 'AIzaSyAECC4ASkB6uOZMQXkbiNYqsaHRGf1_K6k'
HARDCODED_PROGRAMMABLE_SEARCH = '848db169675dd476e'

# --- Location Extraction ---
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


# --- Load Multimodal Data (Adaptation for local paths and no Colab file upload) ---
# The Colab file upload part needs to be removed or commented out.
# File handling needs to be adapted for local paths.

def load_multimodal_data_from_directory(data_directory):
    """
    Loads all files from the specified directory, determines their type by extension,
    and prepares them for further processing. No subfolders required.
    """
    print(f"Loading data from directory: {data_directory}")

    multimodal_data = []

    # Supported file extensions for each modality
    text_exts = {'.txt'}
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    audio_exts = {'.mp3', '.wav', '.aac', '.flac'}
    pdf_exts = {'.pdf'}

    if not os.path.isdir(data_directory):
        print(f"Error: Data directory not found at {data_directory}. Please check the path and ensure it exists.")
        return multimodal_data

    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if not os.path.isfile(filepath):
            continue

        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext in text_exts:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                multimodal_data.append({'id': f'text_{filename}', 'type': 'text', 'text': text_content})

            elif ext in pdf_exts:
                try:
                    extracted_text = extract_text(filepath)
                    multimodal_data.append({'id': f'pdf_{filename}', 'type': 'text', 'text': extracted_text})
                    print(f"Extracted text from PDF '{filename}'")
                except Exception as e:
                    print(f"Error extracting text from PDF {filename}: {e}")

            elif ext in image_exts:
                try:
                    img = Image.open(filepath)
                    multimodal_data.append({'id': f'image_{filename}', 'type': 'image', 'image': img})
                except Exception as e:
                    print(f"Error loading image file {filename}: {e}")

            elif ext in video_exts:
                multimodal_data.append({'id': f'video_{filename}', 'type': 'video', 'video_path': filepath})

            elif ext in audio_exts:
                multimodal_data.append({'id': f'audio_{filename}', 'type': 'audio', 'audio_path': filepath})

            else:
                print(f"Unsupported file type for '{filename}', skipping.")

        except Exception as e:
            print(f"Error processing file '{filename}': {e}")

    print("\n--- Loaded Data Summary ---")
    print(f"Loaded {len([d for d in multimodal_data if d['type']=='text'])} text documents.")
    print(f"Loaded {len([d for d in multimodal_data if d['type']=='image'])} images.")
    print(f"Loaded {len([d for d in multimodal_data if d['type']=='video'])} videos.")
    print(f"Loaded {len([d for d in multimodal_data if d['type']=='audio'])} audio files.")
    print(f"\nTotal items in multimodal_data list: {len(multimodal_data)}")


    return multimodal_data


# --- Process Multimodal Data ---

# This section will be executed after loading data in the main execution block

# Define parameters for video frame extraction (e.g., frames per second)
FRAMES_PER_SECOND = 1 # Extract 1 frame per second

# Define parameters for audio processing (e.g., segment length in milliseconds)
AUDIO_SEGMENT_LENGTH_MS = 5000 # Process audio in 5-second segments


# --- Generate Multimodal Embeddings ---

# Define the Multimodal Embedding Model Class (Ensuring only one definition)
class MultimodalEmbeddingModel:
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        """Initializes the CLIP and OpenL3 models for multimodal embedding generation."""
        # Load pre-trained CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        # Load OpenL3 model for audio embeddings
        self.openl3_model = openl3.models.load_audio_embedding_model(
            input_repr="mel128", content_type="env", embedding_size=512
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        print(f"Using device: {self.device}")


    def get_embedding(self, data, data_type):
        """
        Generates an embedding for a given data item based on its type.

        Args:
            data: The content (text string, PIL Image, or file path).
            data_type: The type of data ('text', 'image', or 'audio').

        Returns:
            A NumPy array representing the embedding.
        """
        if data_type == 'text':
            inputs = self.clip_processor(text=data, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            return text_features.cpu().numpy()

        elif data_type == 'image':
            # Handle single PIL image or a list of images (e.g., from a video)
            inputs = self.clip_processor(images=data, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.cpu().numpy()

        elif data_type == 'audio':
            # OpenL3 requires a numpy array of audio samples and sample rate
            # Assuming 'data' is the audio path for OpenL3
            if isinstance(data, str) and os.path.exists(data):
                 audio, sr = sf.read(data)
                 # OpenL3 takes the entire audio signal, or 1-second segments
                 # We process segments in the processing step, here we get the embedding for the whole audio file
                 # A more refined approach would be to get embeddings per segment and average/pool them.
                 # For now, let's use the whole audio if data is a path.
                 try:
                      emb, ts = openl3.get_audio_embedding(
                          audio, sr, model=self.openl3_model, verbose=False
                      )
                      # If OpenL3 returns multiple embeddings (e.g., for segments), average them
                      if emb.ndim > 1:
                          return np.mean(emb, axis=0)
                      return emb # Return the embedding if it's a single one
                 except Exception as openl3_e:
                      print(f"Error generating OpenL3 embedding for {data}: {openl3_e}")
                      return None # Return None if embedding fails


            else:
                 print(f"Warning: Invalid data type or path for audio embedding: {type(data)}")
                 return None # Return None for invalid audio data


        else:
            raise ValueError(f"Unsupported data type: {data_type}")

# Define the SimpleAIAgent class
class SimpleAIAgent:
    """
    An AI agent capable of understanding user intent, deciding on actions,
    and orchestrating a multimodal RAG workflow.
    """
    def __init__(self, name, embedding_model, tokenizer, processor, modality_indexes, modality_data_items, text_feature_dim, image_feature_dim, audio_feature_dim):
        """
         Initializes the SimpleAIAgent with necessary components for RAG and tool use.

        Args:
            name: The name of the agent.
            embedding_model: The model used for generating embeddings.
            tokenizer: The tokenizer for the embedding model.
            processor: The processor for the embedding model.
            modality_indexes: Dictionary of FAISS indexes for each modality.
            modality_data_items: Dictionary of data items corresponding to each modality's index.
            text_feature_dim: Dimension of text embeddings.
            image_feature_dim: Dimension of image embeddings.
            audio_feature_dim: Dimension of audio embeddings.
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
                available_models = [m.name for m in genai.list_models() if isinstance(m, genai.types.Model) and 'generateContent' in m.supported_generation_methods]
                if available_models:
                     self.gemini_configured = True
                     print(f"Gemini API configured successfully for agent '{self.name}'.")
                else:
                     print(f"Gemini API key found and configured for agent '{self.name}', but no models supporting generateContent were listed. Response generation may fail.")
            else:
                print(f"Gemini API key '{api_key_name}' not found in environment variables for agent '{self.name}'. Response generation will use a placeholder.")
        except Exception as e:
            print(f"Error during Gemini API configuration for agent '{self.name}': {e}. Response generation will use a placeholder.")


    def understand_intent(self, query: str):
        """
        Analyzes the user query to determine the user's intent.

        Args:
            query: The user's query string.

        Returns:
            A string representing the identified intent (e.g., 'perform_rag',
            'display_map', 'calculate_travel_time', 'search_images', 'respond_general').
        """
        # Process query for location entities first
        doc = nlp(query.lower())
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        print(f"Debug: spaCy found {len(locations)} locations: {locations}") # Debug print

        # Prioritize travel time/route intent if two locations are clearly indicated
        travel_keywords = ["travel time", "how long to get to", "route from", "directions from", "distance from", "get from", "to"]
        if any(keyword in query.lower() for keyword in travel_keywords) or len(locations) >= 2:
             print("Debug: Identified potential travel time/route intent keywords or found two locations.") # Debug print
             source, destination = self.extract_source_destination_from_query(query)
             if source and destination:
                  print("Debug: Successfully extracted source and destination for travel time intent.") # Debug print
                  return "calculate_travel_time"
             else:
                  print("Debug: Could not extract two locations for travel time intent. Falling back to next intent check.") # Debug print


        # Check for image search intent keywords
        image_keywords = ["show me images of", "images of", "pictures of", "show me pictures of", "find images of", "find pictures of"]
        if any(keyword in query.lower() for keyword in image_keywords):
             print("Debug: Identified image search intent keywords.") # Debug print
             return "search_images"

        # Check for map display intent keywords or if a single location is mentioned without other specific intents
        map_keywords = ["map", "show map of", "where is"]
        if any(keyword in query.lower() for keyword in map_keywords) or (len(locations) == 1 and not any(keyword in query.lower() for keyword in travel_keywords + image_keywords)):
            print("Debug: Identified map intent keywords or found a single location without other specific intents.") # Debug print
            location = self.extract_location_from_query(query)
            if location:
                print(f"Debug: Extracted single location '{location}', confirming map display intent.")
                return "display_map"
            else:
                print("Debug: Map keyword found but no single location extracted. Falling back to RAG.")


        # Default to RAG if no specific tool intent is detected
        print("Debug: No specific intent keywords or clear location-based intent detected. Defaulting to RAG.") # Debug print
        return "perform_rag"


    def decide_action(self, intent: str):
        """
        Decides the appropriate action based on the identified intent.

        Args:
            intent: The identified intent string.

        Returns:
            A string representing the chosen action ('perform_rag',
            'display_map', 'calculate_travel_time', 'search_images', 'respond_general').
        """
        # Simple mapping of intent to action for now
        # More complex logic could involve checking available tools, data, etc.
        if intent in ["display_map", "calculate_travel_time", "search_images"]:
            return intent
        else:
            return "perform_rag" # Default to RAG for all other intents


    def extract_location_from_query(self, query: str):
        """
        Extracts a location name from the user query.

        Args:
            query: The user's query string.

        Returns:
            A string representing the extracted location name, or None if not found.
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
        match = re.search(r'(?:map of|where is|show map of)\s+(.+)', query, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            print(f"Debug: Regex matched map pattern in extract_location_from_query, extracted: {extracted}") # Debug print
            return extracted
        # Add a pattern to capture just a single location name if no other keywords are present
        match = re.search(r'^\s*([^,]+?)\s*$', query) # Captures text from the start to end if no commas
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
        Extracts source and destination locations from a travel time/route query.

        Args:
            query: The user's query string.

        Returns:
            A tuple containing (source_location, destination_location), or (None, None) if not found.
        """
        doc = nlp(query)
        # Look for named entities that are locations (GPE, LOC)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        print(f"Debug: spaCy found {len(locations)} locations in extract_source_destination_from_query: {locations}") # Debug print

        # Prioritize regex patterns that explicitly mention source and destination
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
            query: The user's query string.

        Returns:
            A NumPy array representing the query embedding.
        """
        try:
            inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=77) # CLIP max length is 77
            with torch.no_grad():
                # Get text features from the embedding model
                # Ensure the embedding model has a get_text_features method or use the multimodal one
                if hasattr(self.embedding_model, 'get_text_features'):
                    query_features = self.embedding_model.get_text_features(inputs.input_ids, inputs.attention_mask)
                elif hasattr(self.embedding_model, 'get_embedding'):
                    # Fallback to the general get_embedding if specific text method isn't found
                    query_features = self.embedding_model.get_embedding(query, 'text')
                else:
                     raise AttributeError("Embedding model does not have a suitable text embedding method.")


            if query_features is not None:
                 # Ensure the output is a 1D NumPy array
                 if isinstance(query_features, torch.Tensor):
                      query_features = query_features.squeeze().cpu().numpy()
                 elif isinstance(query_features, np.ndarray):
                      query_features = query_features.squeeze()


                 if query_features.ndim == 1:
                      return query_features
                 else:
                      print(f"Warning: Generated query embedding is not 1D (shape: {query_features.shape}) after squeezing.")
                      return np.zeros(self.text_feature_dim, dtype='float32') # Return zero vector if not 1D


            else:
                 print("Warning: Embedding generation returned None.")
                 return np.zeros(self.text_feature_dim, dtype='float32') # Return zero vector if embedding is None


        except Exception as e:
            print(f"Error generating query embedding: {e}")
            traceback.print_exc()
            # Return a zero vector or handle appropriately if embedding fails
            return np.zeros(self.text_feature_dim, dtype='float32') # Return a zero vector of the expected dimension




    # --- Implement Retrieval ---

    # Define a function to retrieve relevant data (Ensuring only one definition)
    def retrieve_relevant_data(self, query_embedding, query_text: str, k_per_modality: int = 2, relevance_threshold: float = 50.0):
        """
        Retrieves relevant data by searching modality-specific FAISS indexes and includes a web search fallback.

        Args:
            query_embedding: The embedding of the user's query.
            query_text: The original text of the user's query.
            k_per_modality: The number of nearest neighbors to retrieve from each index.
            relevance_threshold: A threshold for considering a local result relevant (lower distance is more relevant).

        Returns:
            A list of relevant data items, potentially including items from the local indexes
            and web search results. Each item will be a dictionary with a 'source' key
            ('local_index' or 'web_search') and a 'data' key containing the retrieved information.
        """
        relevant_items = []

        # 1. Query the relevant FAISS index(es).
        if query_embedding is not None and self.modality_indexes:
            print("Agent searching modality-specific FAISS indexes...")
            query_embedding_reshaped = query_embedding.reshape(1, -1).astype('float32')

            for modality_name, index in self.modality_indexes.items():
                if index and index.ntotal > 0:
                    print(f"Searching {modality_name} index (dimension {index.d})...")
                    try:
                        # Check if query embedding dimension matches the index dimension.
                        if query_embedding_reshaped.shape[-1] != index.d:
                             print(f"Warning: Query embedding dimension ({query_embedding_reshaped.shape[-1]}) does not match actual {modality_name} index dimension ({index.d}). Skipping search for this modality.")
                             continue

                        distances, indices = index.search(query_embedding_reshaped, k_per_modality)

                        print(f"FAISS search found {len(indices[0])} potential neighbors in {modality_name} index.")

                        # 4. Store the search results for each queried modality.
                        # 5. Combine the results from all queried modalities.
                        for i in range(len(indices[0])):
                            idx = indices[0][i]
                            distance = distances[0][i]

                            # Check if the index is valid and within bounds of the data items list
                            if idx != -1 and idx < len(self.modality_data_items.get(modality_name, [])): # Added bounds check for modality_data_items
                                 # Check if the distance is below the relevance threshold
                                 if distance < relevance_threshold:
                                    # Retrieve the actual data item using the index
                                    data_item = self.modality_data_items[modality_name][idx]
                                    relevant_items.append({'source': 'local_index', 'modality': modality_name, 'data': data_item, 'distance': distance})
                                    print(f"Found relevant item in {modality_name} index: {data_item.get('id', 'N/A')} with distance {distance:.4f}")
                                 else:
                                    print(f"Item {self.modality_data_items[modality_name][idx].get('id', 'N/A')} found with distance {distance:.4f} in {modality_name} index, above relevance threshold.")
                            elif idx != -1:
                                print(f"Item from {modality_name} index found with distance {distance:.4f}, above relevance threshold {relevance_threshold}.")
                            else:
                                print(f"Invalid index found in {modality_name} index search results.")


                    except Exception as e:
                        print(f"Error searching {modality_name} index: {e}")
                        traceback.print_exc()
                else:
                    print(f"No index or empty index for modality: {modality_name}. Skipping search.")

            # 6. Sort the combined list of local results by distance.
            relevant_items.sort(key=lambda x: x.get('distance', float('inf')))
            print(f"Combined and sorted {len(relevant_items)} potential local results.")


        # --- Web Search Fallback ---
        # Perform web search if no relevant local items are found after searching all applicable indexes
        if not relevant_items:
            print("No relevant local data found. Performing web search...")
            try:
                # Adaptation: Use environment variables for API keys, fallback to hardcoded
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
                res = service.cse().list(q=query_text, cx=programmable_search_engine_id, num=k_per_modality).execute()

                if 'items' in res:
                    for item in res['items']:
                        relevant_items.append({'source': 'web_search', 'data': item})
                    print(f"Added {len(res['items'])} web search results.")

            except Exception as e:
                print(f"Error during web search: {e}")
                traceback.print_exc()

        return relevant_items


    def generate_response(self, query: str, relevant_items: list):
        """
        Generates a response using a multimodal LLM (Gemini 1.5 Flash) based on the query and retrieved context.

        Args:
            query: The original user query string.
            relevant_items: A list of dictionaries containing retrieved relevant data.

        Returns:
            A string representing the generated response.
        """
        # Prepare content for the Gemini model, including multimodal parts
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
                    gemini_content.append(f"Data ID: {data_item.get('id', 'N/A')}\n")
                    gemini_content.append(f"Modality: {item.get('modality', 'Unknown')}\n") # Include modality in context
                    if 'text' in data_item and data_item['text']:
                        gemini_content.append(f"Text Content: {data_item['text'][:500]}...\n") # Truncate text for context
                    if 'image' in data_item and isinstance(data_item['image'], Image.Image):
                        # Append the actual image object for Gemini to analyze
                        gemini_content.append(data_item['image'])
                        gemini_content.append("\n") # Add a newline after the image
                        print(f"Agent included image '{data_item.get('id', 'N/A')}' in Gemini prompt.")
                    if item.get('modality') == 'video_frame' and 'image' in data_item and isinstance(data_item['image'], Image.Image):
                         # Append the actual video frame image object
                         gemini_content.append(data_item['image'])
                         gemini_content.append("\n") # Add a newline after the video frame
                         print(f"Agent included video frame '{data_item.get('id', 'N/A')}' in Gemini prompt.")
                    # Assuming 'audio_path' is present in the item for audio segments
                    if item.get('modality') == 'audio_segment' and 'audio_path' in data_item:
                         # Note: Gemini's current multimodal capabilities primarily focus on text and images.
                         # Audio data cannot be directly included in the same way. You would need
                         # a separate audio-specific model or a multimodal model that supports audio.
                         # For now, we'll just note its presence in the text context.
                         gemini_content.append("Audio Data: Present (Note: Audio data is not directly processed by this multimodal model)\n")
                         print(f"Agent noted presence of audio data '{data_item.get('id', 'N/A')}' in text context.")

                    if 'distance' in item:
                         gemini_content.append(f"Retrieval Distance: {item['distance']:.4f}\n")

                elif item.get('source') == 'web_search' and isinstance(data_item, dict):
                     if 'title' in data_item: gemini_content.append(f"Web Search Title: {data_item['title']}\n")
                     if 'body' in data_item: gemini_content.append(f"Web Search Snippet: {data_item['body'][:500]}...\n") # Truncate snippet
                     if 'href' in data_item: gemini_content.append(f"Web Search URL: {data_item['href']}\n")

                else:
                    gemini_content.append(f"Agent received unexpected data format for item: {item}\n")

                gemini_content.append("---\n")
        else:
            gemini_content.append("No relevant context found.")

        gemini_content.append(f"\nGenerated Response:")


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

    # --- Map Functionality ---

    # Define a folium-based map display function (Ensuring only one definition)
    def display_map_for_location(self, location_name: str):
        """
        Generates and displays a Folium map centered at a given location.
        In a script, this will save the map to a temporary HTML file and open it in a browser.

        Args:
            location_name: The name of the location to display the map for.
        """
        print(f"Generating map for location: {location_name}")
        try:
            geolocator = Nominatim(user_agent="multimodal_rag_app")
            location_data = geolocator.geocode(location_name)

            if location_data:
                latitude = location_data.latitude
                longitude = location_data.longitude
                print(f"Found coordinates for '{location_name}': Latitude={latitude}, Longitude={longitude}")
            else:
                print(f"Could not find coordinates for '{location_name}'. Using default location (0,0).")
                latitude = 0
                longitude = 0

            m = folium.Map(location=[latitude, longitude], zoom_start=10)
            folium.Marker([latitude, longitude], popup=location_name).add_to(m)

            # Save the map to a temporary HTML file and open it
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                map_path = tmpfile.name
                m.save(map_path)

            print(f"Saved map to temporary file: {map_path}")
            webbrowser.open(f'file://{map_path}') # Open the HTML file in the default browser
            print("Folium map displayed in a browser.")

        except Exception as e:
            print(f"An error occurred while generating or displaying the Folium map: {e}")

    # Define a folium-based travel time and route function (Ensuring only one definition)
    def display_travel_time_and_route(self, source_location: str, destination_location: str):
        """
        Calculates the travel time and displays the route between two locations on a Folium map using OSRM.
        In a script, this will save the map to a temporary HTML file and open it in a browser.


        Args:
            source_location: The starting location.
            destination_location: The ending location.
        """
        geolocator = Nominatim(user_agent="multimodal_rag_app")

        try:
            source_coords = geolocator.geocode(source_location)
            destination_coords = geolocator.geocode(destination_location)

            if not source_coords or not destination_coords:
                print("Could not find coordinates for one or both locations.")
                return

            source_lon, source_lat = source_coords.longitude, source_coords.latitude
            dest_lon, dest_lat = destination_coords.longitude, destination_coords.latitude

            # Construct the OSRM API request URL
            url = f"http://router.project-osrm.org/route/v1/driving/{source_lon},{source_lat};{dest_lon},{dest_lat}?overview=full&geometries=polyline"

            response = requests.get(url)
            response.raise_for_status()
            directions_data = response.json()

            if directions_data['code'] == 'Ok':
                route = directions_data['routes'][0]
                duration_seconds = route['duration']

                hours = math.floor(duration_seconds / 3600)
                minutes = math.floor((duration_seconds % 3600) / 60)
                travel_time = f"{hours} hours and {minutes} minutes"

                map_center = ((source_lat + dest_lat) / 2,
                              (source_lon + dest_lat) / 2)
                m = folium.Map(location=map_center, zoom_start=10)

                folium.Marker([source_lat, source_lon], popup=f"Start: {source_location}").add_to(m)
                folium.Marker([dest_lat, dest_lon], popup=f"End: {destination_location}").add_to(m)

                # Decode the polyline and add it to the map
                points = polyline.decode(route['geometry'])
                folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(m)

                # Save the map to a temporary HTML file and open it
                with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                     map_path = tmpfile.name
                     m.save(map_path)

                print(f"Saved map to temporary file: {map_path}")
                webbrowser.open(f'file://{map_path}') # Open the HTML file in the default browser
                print("Folium map displayed in a browser.")

                print(f"The estimated travel time from {source_location} to {destination_location} is {travel_time}.")

            else:
                print(f"Could not find directions. Status: {directions_data['code']}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    # --- Image Search and Display ---

    # Define search_images and display_images functions (Ensuring only one definition)
    def search_images(self, query: str, num: int = 5):
        """
        Searches for images using the Google Custom Search API.

        Args:
            query: The search query.
            num: The number of images to return.
        """
        # Adaptation: Use environment variables for API keys, fallback to hardcoded
        api_key = os.getenv('GOOGLE_CLOUD_API_KEY')
        cse_id = os.getenv('PROGRAMMABLE_SEARCH')

        # Fallback to hardcoded values if environment variables are not set
        if not api_key:
             api_key = HARDCODED_GOOGLE_CLOUD_API_KEY
             print("Using hardcoded GOOGLE_CLOUD_API_KEY for image search.")
        if not cse_id:
             cse_id = HARDCODED_PROGRAMMABLE_SEARCH
             print("Using hardcoded PROGRAMMABLE_SEARCH ID for image search.")


        if not api_key or not cse_id:
            print("Google Cloud API Key or Programmable Search Engine ID not available (neither from env vars nor hardcoded). Cannot perform image search.")
            return []

        service = build("customsearch", "v1", developerKey=api_key)
        try:
            res = service.cse().list(q=query, cx=cse_id, searchType='image', num=num).execute()
            return [item['link'] for item in res['items']]
        except Exception as e:
            print(f"An error occurred during image search: {e}")
            return []

    def display_images(self, image_urls: list):
        """
        Displays images from a list of URLs using Pillow's show() method.

        Args:
            image_urls: A list of image URLs.
        """
        print(f"Attempting to display {len(image_urls)} images...")
        for url in image_urls:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                # Use Pillow's show() method to open the image in a default viewer
                img.show()
                print(f"Displayed image from URL: {url}")
            except Exception as e:
                print(f"Could not display image from URL {url}. Error: {e}")

    # --- Agent Orchestration and Workflow ---

    def orchestrate_workflow(self, query: str):
        """
        Orchestrates the workflow based on the user's query.

        Args:
            query: The user's query string.
        """
        print(f"\nAgent '{self.name}' received query: '{query}'")

        # 1. Understand Intent
        intent = self.understand_intent(query)
        print(f"Agent identified intent: '{intent}'")

        # 2. Decide Action
        action = self.decide_action(intent)
        print(f"Agent decided action: '{action}'")

        # 3. Execute Action
        if action == "perform_rag":
            # Generate query embedding
            query_embedding = self.generate_query_embedding(query)

            # Retrieve relevant data using modality-specific indexes and data items
            # Pass the modality_indexes and modality_data_items to the retrieve function
            relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)

            # Generate response using LLM
            response = self.generate_response(query, relevant_items)
            print(f"\nAgent Response:\n{response}")

        elif action == "display_map":
            # Extract location
            location = self.extract_location_from_query(query)
            if location:
                 self.display_map_for_location(location)
                 # Optionally, ask if the user wants images as well
                 image_query = f"images of {location}"
                 print(f"Agent considering related image search for: '{image_query}'")
                 # Call search_images and then display_images
                 image_urls = self.search_images(image_query)
                 if image_urls:
                      self.display_images(image_urls)


            else:
                print("Agent could not extract a location from the query for map display.")
                # Fallback to RAG or general response if location extraction fails
                print("Agent falling back to multimodal RAG...")
                query_embedding = self.generate_query_embedding(query)
                # Pass the modality_indexes and modality_data_items to the retrieve function
                relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)
                response = self.generate_response(query, relevant_items)
                print(f"\nAgent Response:\n{response}")


        elif action == "calculate_travel_time":
            # Extract source and destination
            source, destination = self.extract_source_destination_from_query(query)
            if source and destination:
                self.display_travel_time_and_route(source, destination)
            else:
                print("Agent could not extract source and destination locations for travel time calculation.")
                # Fallback to RAG or general response if location extraction fails
                print("Agent falling back to multimodal RAG...")
                query_embedding = self.generate_query_embedding(query)
                # Pass the modality_indexes and modality_data_items to the retrieve function
                relevant_items = self.retrieve_relevant_data(query_embedding, query, relevance_threshold=RELEVANCE_THRESHOLD)
                response = self.generate_response(query, relevant_items)
                print(f"\nAgent Response:\n{response}")


        elif action == "search_images":
             # Extract the subject for image search (simple extraction for now)
             # Use regex to extract the subject more accurately after "show me images of", "images of", etc.
             match = re.search(r'(?:show me |find )?(?:images?|pictures?) of\s+(.+)', query, re.IGNORECASE)
             if match:
                 image_subject = match.group(1).strip()
                 print(f"Debug: Extracted image search subject: '{image_subject}'")
                 # Call search_images and then display_images
                 image_urls = self.search_images(image_subject)
                 if image_urls:
                      self.display_images(image_urls)
                 else:
                      print(f"No images found for '{image_subject}'.")
             else:
                 print("Agent could not identify the subject for image search.")
                 # Fallback to general response
                 print("Agent providing a general response.")
                 response = self.generate_response(query, []) # Generate response without context
                 print(f"\nAgent Response:\n{response}")


        elif action == "respond_general":
            # Handle general queries, potentially using the LLM without specific retrieval
            print("Agent providing a general response.")
            response = self.generate_response(query, []) # Generate response without context
            print(f"\nAgent Response:\n{response}")

        else: # unknown_action
            print("Agent could not determine the action for the query.")
            print("Agent providing a general response.")
            response = self.generate_response(query, []) # Generate response without context
            print(f"\nAgent Response:\n{response}")

        print("-" * 50) # Separator


# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize variables that will be populated during data loading and processing
    multimodal_data = []
    processed_multimodal_data = []
    multimodal_data_with_embeddings = []
    modality_data_items = {}
    modality_embeddings = {}
    modality_indexes = {}
    multimodal_embedding_model = None # Initialize the model to None

    # Instantiate the MultimodalEmbeddingModel
    try:
        # Define dummy dimensions if not already defined
        if 'text_feature_dim' not in locals(): text_feature_dim = 768
        if 'image_feature_dim' not in locals(): image_feature_dim = 768 # Use 768 as output dim for projection
        if 'audio_feature_dim' not in locals(): audio_feature_dim = 512

        # Load tokenizer and processor for the embedding model
        print("Loading tokenizer and processor for embedding model.")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")


        multimodal_embedding_model = MultimodalEmbeddingModel() # Instantiate the actual model
        print("MultimodalEmbeddingModel instantiated.")
    except Exception as e:
        print(f"Error instantiating MultimodalEmbeddingModel: {e}")
        multimodal_embedding_model = None # Ensure it's None if instantiation fails


    # Load data from the specified local directory
    # !!! IMPORTANT: Replace this path with the actual path to your data directory !!!
    data_directory = r"C:\Users\spnes\Downloads\industrial training to do\.venv\multimodal files"


    # Load all files directly from the folder (no subfolders)
    multimodal_data = load_multimodal_data_from_directory(data_directory)

    # Process multimodal data
    print("Processing multimodal data...")
    FRAMES_PER_SECOND = 1
    AUDIO_SEGMENT_LENGTH_MS = 5000

    for item in multimodal_data:
        item_type = item['type']
        item_id = item['id']

        if item_type == 'text':
            if 'text' in item and item['text']:
                 processed_multimodal_data.append(item)
            else:
                print(f"Skipping processing for empty text item {item_id}.")

        elif item_type == 'image':
            if 'image' in item and isinstance(item['image'], Image.Image):
                 processed_multimodal_data.append(item)
            else:
                 print(f"Skipping processing for invalid image item {item_id}.")

        elif item_type == 'video':
            video_path = item['video_path']
            print(f"Processing video file: {video_path}")
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                     print(f"Error: Could not open video file {video_path}")
                     continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_interval = max(int(fps / FRAMES_PER_SECOND), 1) if fps > 0 else 1
                count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % frame_interval == 0:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        processed_multimodal_data.append({'id': f'{item_id}_frame_{count}', 'type': 'video_frame', 'video_id': item_id, 'frame_index': count, 'image': img})
                    count += 1
                cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"Error processing video file {video_path}: {e}")

        elif item_type == 'audio':
            audio_path = item['audio_path']
            print(f"Processing audio file: {audio_path}")
            try:
                audio = AudioSegment.from_file(audio_path)
                duration_ms = len(audio)
                for i in range(0, duration_ms, AUDIO_SEGMENT_LENGTH_MS):
                    segment = audio[i:i + AUDIO_SEGMENT_LENGTH_MS]
                    processed_multimodal_data.append({'id': f'{item_id}_segment_{i}', 'type': 'audio_segment', 'audio_id': item_id, 'start_time_ms': i, 'end_time_ms': i + len(segment), 'audio_path': audio_path, 'audio_segment_data': segment})
            except Exception as e:
                print(f"Error processing audio file {audio_path}: {e}")

    print(f"\nFinished processing multimodal data. Total items for embedding: {len(processed_multimodal_data)}")


    # Generate embeddings
    print(f"Generating embeddings for {len(processed_multimodal_data)} items...")
    # Initialize modality_data_items and modality_embeddings with empty lists based on processed data types
    for item in processed_multimodal_data:
        modality_type = item.get('type')
        if modality_type not in modality_data_items:
             modality_data_items[modality_type] = []
        if modality_type not in modality_embeddings:
             modality_embeddings[modality_type] = []

    if multimodal_embedding_model and processed_multimodal_data:
        for i, item in enumerate(processed_multimodal_data):
            item_id = item.get('id', f'item_{i}')
            item_type = item.get('type')

            try:
                embedding = None
                if item_type == 'text':
                    if 'text' in item and item['text']:
                        embedding = multimodal_embedding_model.get_embedding(item['text'], 'text')

                elif item_type == 'image':
                     if 'image' in item and isinstance(item['image'], Image.Image):
                        embedding = multimodal_embedding_model.get_embedding(item['image'], 'image')

                elif item_type == 'video_frame':
                     if 'image' in item and isinstance(item['image'], Image.Image):
                        embedding = multimodal_embedding_model.get_embedding(item['image'], 'image')

                elif item_type == 'audio_segment':
                     # For audio segments, pass the path to the embedding model
                     if 'audio_path' in item and os.path.exists(item['audio_path']):
                          try:
                             embedding = multimodal_embedding_model.get_embedding(item['audio_path'], 'audio')
                          except Exception as audio_e:
                              print(f"Error generating audio embedding for {item_id} from path {item['audio_path']}: {audio_e}")
                              embedding = None
                     else:
                         print(f"Skipping embedding for audio segment {item_id}: audio path is missing or invalid.")
                         embedding = None

                if embedding is not None:
                    if embedding.ndim == 2 and embedding.shape[0] == 1:
                        embedding = embedding.squeeze()
                    if embedding.ndim == 1:
                        item['embedding'] = embedding
                        multimodal_data_with_embeddings.append(item)
                        if item_type not in modality_data_items:
                            modality_data_items[item_type] = []
                        modality_data_items[item_type].append(item)
                        if item_type not in modality_embeddings:
                            modality_embeddings[item_type] = []
                        modality_embeddings[item_type].append(embedding)
                    else:
                        print(f"Warning: Embedding for item {item_id} ({item_type}) is not 1D after processing (shape: {embedding.shape}). Skipping.")

            except Exception as e:
                print(f"Error generating embedding for item {item_id} ({item_type}): {e}")


        print(f"\nFinished generating embeddings. Successfully generated {len(multimodal_data_with_embeddings)} embeddings.")
        print(f"Stored {len(multimodal_data_with_embeddings)} data items with embeddings.")
        print(f"Populated modality_data_items for modalities: {list(modality_data_items.keys())}")
        print(f"Organized embeddings for modalities: {list(modality_embeddings.keys())}")
    else:
        print("MultimodalEmbeddingModel is not initialized or processed_multimodal_data is empty. Cannot generate embeddings or build modality_embeddings.")


    # Build FAISS indexes
    modality_indexes = {}
    if modality_embeddings:
        print("Building FAISS indexes for each modality...")
        for modality, embeddings_list in modality_embeddings.items():
            if embeddings_list:
                try:
                    embeddings_np = np.array(embeddings_list).astype('float32')
                    embedding_dimension = embeddings_np.shape[-1]
                    print(f"Building FAISS index for {modality} with dimension {embedding_dimension}...")
                    index = faiss.IndexFlatL2(embedding_dimension)
                    index.add(embeddings_np)
                    modality_indexes[modality] = index
                    print(f"FAISS index built for {modality} with {index.ntotal} embeddings.")
                except Exception as e:
                    print(f"An error occurred while building the FAISS index for {modality}: {e}")
            else:
                print(f"No embeddings available for modality: {modality}. Skipping index creation.")
    else:
        print("No modality embeddings available. Cannot build FAISS indexes.")


    # Check if the Gemini API is successfully configured once before the loop
    gemini_configured = False
    try:
        google_api_key_for_config = os.getenv('NEW_GOOGLE_API_KEY') # Use a separate variable for initial config check
        if google_api_key_for_config is None:
            print("Warning: Gemini API key (NEW_GOOGLE_API_KEY) not found in environment variables. Response generation will use a placeholder.")
        else:
            genai.configure(api_key=google_api_key_for_config)
            gemini_configured = True
            print("Gemini API is configured and will be used for response generation.")
    except Exception as e:
        print(f"An unexpected error occurred during Gemini API configuration (NEW_GOOGLE_API_KEY from env): {e}. Response generation will use a placeholder.")


    # Instantiate the SimpleAIAgent with all necessary components
    print("\nInstantiating SimpleAIAgent...")
    # Ensure necessary variables for agent instantiation are defined or have default values
    if 'text_feature_dim' not in locals(): text_feature_dim = 768
    if 'image_feature_dim' not in locals(): image_feature_dim = 768 # Use 768 as output dim for projection
    if 'audio_feature_dim' not in locals(): audio_feature_dim = 512

    # Load tokenizer and processor for the embedding model if not already loaded
    if 'tokenizer' not in locals():
         print("Loading tokenizer for agent instantiation.")
         tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    if 'processor' not in locals():
         print("Loading processor for agent instantiation.")
         processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Pass the modality_indexes and modality_data_items
    agent = SimpleAIAgent(
        name="MultimodalAgent",
        embedding_model=multimodal_embedding_model,
        tokenizer=tokenizer,
        processor=processor,
        modality_indexes=modality_indexes, # Pass the modality-specific indexes
        modality_data_items=modality_data_items, # Pass the modality-specific data items
        text_feature_dim=text_feature_dim,
        image_feature_dim=image_feature_dim,
        audio_feature_dim=audio_feature_dim
    )
    print(f"SimpleAIAgent '{agent.name}' instantiated.")


    # Define a relevance threshold
    RELEVANCE_THRESHOLD = 40.0 # Changed threshold to 40.0

    # Implement the interactive query loop
    print("--- Multimodal RAG System Ready ---")
    print("Enter your queries to search the multimodal data, or ask about a location to see a map or travel route.")
    print("Type 'quit' to exit the query loop.")

    while True:
        query = input("Enter your query: ")

        if query.lower() == 'quit':
            print("Exiting query loop.")
            break

        # Delegate the query handling to the agent's orchestration workflow
        agent.orchestrate_workflow(query)

        print("-" * 50) # Separator
