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
# from google.colab import files # Import files for Colab file upload - Removed for local environment


#Model Context Protocol legal document summarization in progress
def summarize_via_mcp(text):
    try:
        response = requests.post("http://localhost:3000/upload", json={"text": text}, timeout=30)
        response.raise_for_status()
        return response.json().get("summary", "No summary returned.")
    except Exception as e:
        return f"Error while summarizing: {e}"

# The following line is for demonstration purposes and assumes a function retrieve_document_from_vector_db exists
# retrieved_doc = retrieve_document_from_vector_db("nda_contract") # This function is not defined in the provided code, but I'm keeping it as is.
# summary = summarize_via_mcp(retrieved_doc)
# print("Summary for user:", summary)


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

# --- Load Multimodal Data (Adaptation for local paths and no Colab file upload) ---
# The Colab file upload part needs to be removed or commented out.
# File handling needs to be adapted for local paths.

# Remove individual load functions for subdirectories
# def load_text_files(directory_path):
#     """Loads text files from a directory."""
#     ...

# def load_image_files(directory_path):
#     """Loads image files from a directory."""
#     ...

# def load_video_files(directory_path):
#     """Loads video file paths from a directory."""
#     ...

# def load_audio_files(directory_path):
#     """Loads audio file paths from a directory."""
#     ...


# Handle file uploads functionality is removed for script
# def handle_uploads_multiple():
#     """Handles multiple file uploads in a Colab environment until user inputs 'done'."""
#     # ... (removed)


# Create temporary directories to save uploaded files for processing (Removed/Adapted)
# Adaptation: Define a single function to load all supported files from a given directory
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


# --- Build FAISS Index ---

# This section will be executed after embedding generation in the main execution block


# --- Implement Retrieval ---

# Define a function to retrieve relevant data (Ensuring only one definition)
def retrieve_relevant_data_with_fallback(query_text: str, multimodal_embedding_model, modality_indexes: dict, modality_data_items: dict, k_per_modality: int = 2, relevance_threshold: float = 50.0):
    """
    Retrieves relevant data by searching modality-specific FAISS indexes and includes a web search fallback.

    Args:
        query_text: The original text of the user's query.
        multimodal_embedding_model: The initialized MultimodalEmbeddingModel instance.
        modality_indexes: A dictionary of modality names to FAISS indexes.
        modality_data_items: A dictionary of modality names to lists of data items.
        k_per_modality: The number of nearest neighbors to retrieve from each modality's index.
        relevance_threshold: A threshold for considering a local result relevant (lower distance is more relevant).

    Returns:
        A list of relevant data items, potentially including items from the local indexes
        and web search results. Each item will be a dictionary with a 'source' key
        ('local_index' or 'web_search') and a 'data' key containing the retrieved information.
    """
    relevant_items = []

    # 1. Generate the query embedding(s) based on the query type.
    query_embedding = None
    if multimodal_embedding_model:
        try:
            print(f"Generating text embedding for query: '{query_text}'")
            query_embedding = multimodal_embedding_model.get_embedding(query_text, 'text')

            if query_embedding is not None:
                if query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
                    query_embedding = query_embedding.squeeze()
                if query_embedding.ndim == 1:
                    print(f"Generated query embedding with shape: {query_embedding.shape}")
                else:
                    print(f"Warning: Generated query embedding is not 1D (shape: {query_embedding.shape}). Cannot use for FAISS search.")
                    query_embedding = None # Invalidate if not 1D

        except Exception as e:
            print(f"Error generating query embedding: {e}")
            query_embedding = None
    else:
        print("MultimodalEmbeddingModel is not available. Cannot generate query embedding.")


    # 2. Query the relevant FAISS index(es).
    if query_embedding is not None and modality_indexes:
        print("Searching modality-specific FAISS indexes...")
        query_embedding_reshaped = query_embedding.reshape(1, -1).astype('float32')

        for modality, index in modality_indexes.items():
            if index and index.ntotal > 0:
                print(f"Searching {modality} index (dimension {index.d})...")
                try:
                    # Check if the query embedding dimension matches the index dimension.
                    if query_embedding_reshaped.shape[-1] != index.d:
                         print(f"Warning: Query embedding dimension ({query_embedding_reshaped.shape[-1]}) does not match actual {modality} index dimension ({index.d}). Skipping search for this modality.")
                         continue

                    # Query the index
                    distances, indices = index.search(query_embedding_reshaped, k_per_modality)

                    print(f"FAISS search found {len(indices[0])} potential neighbors in {modality} index.")

                    # 4. Store the search results for each queried modality.
                    # 5. Combine the results from all queried modalities.
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        distance = distances[0][i]

                        # Check if the index is valid and the distance is below the relevance threshold
                        if idx != -1 and distance < relevance_threshold:
                             # Retrieve the actual data item using the index from the correct modality data list
                             # We need to map the index from the modality-specific index back to the correct item
                             # in the modality_data_items[modality] list.
                             # Since modality_data_items[modality] was built in the same order as embeddings were added to the modality index,
                             # the index in the FAISS result corresponds to the index in modality_data_items[modality].
                            if modality in modality_data_items and idx < len(modality_data_items[modality]):
                                data_item = modality_data_items[modality][idx]
                                # Add modality information and distance to the relevant item
                                relevant_items.append({'source': 'local_index', 'modality': modality, 'data': data_item, 'distance': distance})
                                print(f"Found relevant item in {modality} index: {data_item.get('id', 'N/A')} with distance {distance:.4f}")
                            else:
                                print(f"Warning: Could not retrieve data item for index {idx} from {modality} data list.")
                        elif idx != -1 and distance >= relevance_threshold:
                            print(f"Item from {modality} index found with distance {distance:.4f}, above relevance threshold {relevance_threshold}.")
                        else:
                            print(f"Invalid index found in {modality} index search results.")


                except Exception as e:
                    print(f"Error during search of {modality} index: {e}")
            else:
                print(f"No index or empty index for modality: {modality}. Skipping search.")

        # 6. Sort the combined list of local results by distance.
        relevant_items.sort(key=lambda x: x.get('distance', float('inf')))
        print(f"Combined and sorted {len(relevant_items)} potential local results.")

        # 7. Filtering based on relevance_threshold is already done in the loop

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
            res = service.cse().list(q=query_text, cx=programmable_search_engine_id).execute()

            if 'items' in res:
                for item in res['items']:
                    relevant_items.append({'source': 'web_search', 'data': item})
                print(f"Added {len(res['items'])} web search results.")

        except Exception as e:
            print(f"Error during web search: {e}")

    return relevant_items


# --- Generate Response (Gemini Integration) ---

# Define generate_response_with_gemini function (Ensuring only one definition)
def generate_response_with_gemini(query: str, relevant_items: list):
    """
    Generates a response using a multimodal LLM (Gemini 1.5 Flash) based on the query and retrieved context.

    Args:
        query: The original user query string.
        relevant_items: A list of dictionaries containing retrieved relevant data.

    Returns:
        A string representing the generated response.
    """
    # Construct the context string to pass to the LLM
    context_text = "Retrieved Context:\n"

    if relevant_items:
        for i, item in enumerate(relevant_items):
            context_text += f"--- Item {i+1} (Source: {item.get('source', 'Unknown')}) ---\n"
            data_item = item.get('data', {})

            if item.get('source') == 'local_index' and isinstance(data_item, dict):
                context_text += f"Data ID: {data_item.get('id', 'N/A')}\n"
                context_text += f"Modality: {item.get('modality', 'Unknown')}\n" # Include modality in context
                if 'text' in data_item and data_item['text']:
                    context_text += f"Text Content: {data_item['text'][:500]}...\n" # Truncate text for context
                if 'image' in data_item and isinstance(data_item['image'], Image.Image):
                    # For Gemini, you would typically pass the image object directly
                    # but for constructing the text context, we just note its presence.
                    context_text += "Image: Present\n"
                if item.get('modality') == 'video_frame' and 'image' in data_item and isinstance(data_item['image'], Image.Image):
                     context_text += "Video Frame: Present\n"
                # Assuming 'audio_path' is present in the item for audio segments
                if item.get('modality') == 'audio_segment' and 'audio_path' in data_item:
                     context_text += f"Audio Segment from file: {os.path.basename(data_item['audio_path'])}\n" # Indicate audio presence

                if 'distance' in item:
                     context_text += f"Distance: {item['distance']:.4f}\n"

            elif item.get('source') == 'web_search' and isinstance(data_item, dict):
                 if 'title' in data_item: context_text += f"Title: {data_item['title']}\n"
                 if 'body' in data_item: context_text += f"Snippet: {data_item['body'][:500]}...\n" # Truncate snippet
                 if 'href' in data_item: context_text += f"URL: {data_item['href']}\n"

            else:
                context_text += f"Unexpected data format for item: {item}\n"

            context_text += "---\n"
    else:
        context_text += "No relevant context found."

    # --- Integrate Gemini API for Response Generation ---
    # Use environment variables for API key in a local environment
    # Retrieve API key here within the function to be safe, although global config is attempted
    google_api_key = os.getenv('NEW_GOOGLE_API_KEY') # Get the API key from environment variables

    if google_api_key is None:
        print("Warning: Gemini API key (NEW_GOOGLE_API_KEY) not found in environment variables. Cannot generate response.")
        return f"Gemini API key (NEW_GOOGLE_API_KEY) not configured. Cannot generate response.\n\n{context_text}"

    try:
        # Initialize the Gemini model
        model_name = 'gemini-1.5-flash-latest'
        model = None # Initialize model to None before finding and initializing

        # Check if the requested model is available and supports generateContent
        # Re-list models here in case the global list is outdated or configuration changed
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]

        if model_name in available_models:
            model = genai.GenerativeModel(model_name)
            print(f"Initialized model: {model_name}")
        else:
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
             prompt_parts = [
                 f"Given the following query and context, generate a concise and helpful response. Focus on the information provided in the context. If the context does not contain enough information to fully answer the query, acknowledge this.\n\n",
                 f"Query: {query}\n\n",
                 context_text,
                 f"\nResponse:"
             ]

             # Generate content using the Gemini model
             # Handle potential errors during generation
             try:
                 gemini_response = model.generate_content(prompt_parts)
                 return gemini_response.text
             except Exception as e:
                 print(f"Error during model.generate_content: {e}")
                 return f"Error generating response from Gemini API: {e}\n\n{context_text}"

        else:
            # Return placeholder if no model could be initialized
            return f"Cannot generate response because no suitable Gemini Flash model was found or initialized.\n\n{context_text}"


    except Exception as e:
        return f"Error calling Gemini API during response generation setup: {e}\n\n{context_text}"


# --- Map Functionality ---

# Define a folium-based map display function (Ensuring only one definition)
def display_map_for_location(location_name: str):
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
def display_travel_time_and_route(source_location: str, destination_location: str):
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
def search_images(query: str, num: int = 5):
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

def display_images(image_urls: list):
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

def extract_location_from_query(query: str) -> str | None:
    """
    Extracts a location name from a user query using spaCy NER.

    Args:
        query: The user's query string.

    Returns:
        The identified location string, or None if no location is found.
    """
    doc = nlp(query)
    location = None
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "ORG", "FAC"]: # Added FAC (Facilities)
            location = ent.text
            print(f"Identified potential location: {location} (Label: {ent.label_})")
            return location

    # Fallback: If no specific entity is found, treat the whole query as a potential location
    # This might lead to inaccurate geocoding, but matches the original notebook's fallback logic.
    if not location and query.strip(): # Check if query is not just whitespace
         print("No specific location entity identified, treating the whole query as a potential location.")
         return query.strip()


    print("No specific location identified in the query.")
    return None


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
        multimodal_embedding_model = MultimodalEmbeddingModel()
        print("MultimodalEmbeddingModel instantiated.")
    except Exception as e:
        print(f"Error instantiating MultimodalEmbeddingModel: {e}")
        multimodal_embedding_model = None # Ensure it's None if instantiation fails


    # Load data from the specified local directory
    data_directory = os.getenv('MULTIMODAL_DATA_DIRECTORY')

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

    # Define a relevance threshold
    RELEVANCE_THRESHOLD = 40.0 # Changed threshold to 40.0

    # Define the default directory for legal document uploads
    LEGAL_DOC_UPLOAD_DIR = os.getenv('LEGAL_DOC_UPLOAD_DIR')

    # Implement the interactive query loop
    print("--- Multimodal RAG System Ready ---")
    print("Enter your queries to search the multimodal data, ask about a location to see a map, or type 'summarize' to summarize a legal document from the upload directory.")
    print("Type 'quit' to exit the query loop.")

    while True:
        query = input("Enter your query: ")

        if query.lower() == 'quit':
            print("Exiting query loop.")
            break

        # Ask user for search type, now including 'legal' option
        search_type_input = input("Perform 'location', 'multimodal', or 'legal' document summarization search? ").lower()

        if search_type_input == 'legal':
            print(f"Please enter the filename of the legal document in the '{LEGAL_DOC_UPLOAD_DIR}' directory you want to summarize.")
            file_name = input("Document filename: ")
            file_path = os.path.join(LEGAL_DOC_UPLOAD_DIR, file_name)

            if os.path.exists(file_path):
                print(f'Attempting to summarize file: {file_path}')

                file_extension = os.path.splitext(file_path)[1].lower()
                document_text = ""

                try:
                    if file_extension == '.pdf':
                        document_text = extract_text(file_path)
                    elif file_extension == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            document_text = f.read()
                    else:
                        print(f"Unsupported file type for summarization: {file_extension}. Please provide a path to a PDF or TXT file.")
                        continue

                    if document_text:
                        print("Summarizing document...")
                        summary = summarize_via_mcp(document_text)
                        print("\n--- Document Summary ---")
                        print(summary)
                        print("------------------------")
                    else:
                        print("Could not extract text from the document.")

                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            else:
                print("Error: File not found at the specified path.")

            # Do not trigger any other reaction
            continue

        elif search_type_input == 'location':
            extracted_location = None
            try:
                extracted_location = extract_location_from_query(query)
            except Exception as e:
                print(f"Error during location extraction: {e}")

            if extracted_location:
                print(f"Location '{extracted_location}' identified in the query.")
                image_urls = search_images(extracted_location)
                if image_urls:
                    print(f"Displaying {len(image_urls)} images for '{extracted_location}':")
                    display_images(image_urls)
                else:
                    print(f"No images found for '{extracted_location}'.")

                try:
                    display_map_for_location(extracted_location)
                except Exception as e:
                    print(f"Error displaying map for location '{extracted_location}': {e}")

                source_location = input("Please enter your source location to calculate travel time: ")
                if source_location:
                    try:
                        display_travel_time_and_route(source_location, extracted_location)
                    except Exception as e:
                        print(f"Error calculating and displaying travel time and route: {e}")
                else:
                    print("No source location provided. Skipping travel time calculation.")

            else:
                print("Could not extract a location from your query.")

        elif search_type_input == 'multimodal':
            relevant_items = []
            if multimodal_embedding_model and modality_indexes and modality_data_items:
                try:
                    print("Performing multimodal search...")
                    relevant_items = retrieve_relevant_data_with_fallback(query, multimodal_embedding_model, modality_indexes, modality_data_items, k_per_modality=3, relevance_threshold=RELEVANCE_THRESHOLD)
                    print(f"Retrieved {len(relevant_items)} relevant items.")
                except Exception as e:
                    print(f"Error during multimodal retrieval: {e}")
                    relevant_items = []
            else:
                print("Warning: Multimodal embedding model, modality indexes, or data items are not available. Skipping local multimodal retrieval.")
                try:
                     print("Attempting web search fallback...")
                     relevant_items = retrieve_relevant_data_with_fallback(query, None, {}, {}, k_per_modality=0, relevance_threshold=RELEVANCE_THRESHOLD)
                     print(f"Retrieved {len(relevant_items)} items from fallback.")
                except Exception as e:
                     print(f"Error during web search fallback: {e}")
                     relevant_items = []

            # Check if generate_response_with_gemini is defined and Gemini is configured
            if 'generate_response_with_gemini' in locals():
                # Use the gemini_configured flag set before the loop
                # The generate_response_with_gemini function itself handles the case where API is not configured.
                generated_response = generate_response_with_gemini(query, relevant_items)
            else:
                 print("Error: 'generate_response_with_gemini' function not found. Cannot generate response.")
                 generated_response = "Error: Response generation function not available."


            print(f"Original Query: {query}")
            print("Retrieved Data Summary:")
            print(f"Generated Response:\n{generated_response}")
        else:
            print("Invalid search type. Please enter 'location', 'multimodal', or 'legal'.")

        print("-" * 50)
