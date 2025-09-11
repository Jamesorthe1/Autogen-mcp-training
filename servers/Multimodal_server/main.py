import os
import numpy as np
import faiss
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoProcessor
import google.generativeai as genai
import traceback
from PIL import Image # Import the Image class
import torch # Import torch

# Import components from other files
from data_processing import load_multimodal_data_from_directory, process_multimodal_data
from embedding_model import MultimodalEmbeddingModel
from ai_agent import SimpleAIAgent # The agent includes tool calls internally

# Load environment variables from .env file using a more robust path
# This assumes the .env file is in the parent directory of the script's location
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), ".env")))

# --- Configuration and Initialization ---

# Check if the Gemini API is successfully configured once before the loop
# This configuration is essential for the agent's response generation.
gemini_configured = False
try:
    google_api_key_for_config = os.getenv('NEW_GOOGLE_API_KEY') # Use the secret name from your .env file
    if google_api_key_for_config is None:
        print("Warning: Gemini API key (NEW_GOOGLE_API_KEY) not found in environment variables. Response generation will use a placeholder.")
    else:
        genai.configure(api_key=google_api_key_for_config)
        # Verify configuration by listing models supporting generateContent
        try:
            available_models = [m.name for m in genai.list_models() if isinstance(m, genai.types.Model) and 'generateContent' in m.supported_generation_methods]
            if available_models:
                 gemini_configured = True
                 print("Gemini API is configured and will be used for response generation.")
                 print(f"Available models supporting generateContent: {available_models}")
            else:
                 print("Warning: Gemini API key found and configured, but no models supporting generateContent were listed. Response generation may fail.")
                 gemini_configured = False # Explicitly set to False if no models found
        except Exception as e:
            print(f"Error listing Gemini models: {e}. Response generation may fail.")
            gemini_configured = False # Explicitly set to False on error

except Exception as e:
    print(f"An unexpected error occurred during Gemini API configuration (checking NEW_GOOGLE_API_KEY from env): {e}. Response generation will use a placeholder.")
    gemini_configured = False # Explicitly set to False on error


# Define dimensions for embeddings (should match the embedding model output)
# These are examples; the actual dimensions depend on the specific models used.
TEXT_FEATURE_DIM = 768 # Example dimension for CLIP text embeddings
IMAGE_FEATURE_DIM = 768 # Example dimension for CLIP image embeddings
AUDIO_FEATURE_DIM = 512 # Example dimension for OpenL3 audio embeddings


# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize variables that will be populated during data loading and processing
    multimodal_data = [] # Stores raw data loaded from directory
    processed_multimodal_data = [] # Stores processed data (e.g., video frames, audio segments)
    multimodal_data_with_embeddings = [] # Stores processed data with generated embeddings
    modality_data_items = {} # Dictionary to hold processed data items organized by modality type
    modality_embeddings = {} # Dictionary to hold embeddings organized by modality type
    modality_indexes = {} # Dictionary to hold FAISS indexes for each modality
    multimodal_embedding_model = None # Instance of the embedding model

    # Instantiate the MultimodalEmbeddingModel
    try:
        print("\nInstantiating MultimodalEmbeddingModel...")
        # Load tokenizer and processor for the embedding model (e.g., CLIP)
        # These are needed for the Agent as well, so load them here and pass.
        print("Loading tokenizer and processor for embedding model.")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Tokenizer and processor loaded.")

        # Instantiate the actual multimodal embedding model
        multimodal_embedding_model = MultimodalEmbeddingModel()
        print("MultimodalEmbeddingModel instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating MultimodalEmbeddingModel: {e}")
        traceback.print_exc()
        multimodal_embedding_model = None # Ensure it's None if instantiation fails
        tokenizer = None # Ensure these are None as well if model loading fails
        processor = None


    # Load data from the specified local directory from environment variable
    data_directory = os.getenv("MULTIMODAL_DATA_DIRECTORY")

    # !!! IMPORTANT: Ensure the MULTIMODAL_DATA_DIRECTORY environment variable is set in your .env file
    #                and the path is correct for your VS Code environment.
    if not data_directory:
        print("Error: MULTIMODAL_DATA_DIRECTORY environment variable is not set.")
        print("Please add MULTIMODAL_DATA_DIRECTORY=<path_to_your_multimodal_files> to your .env file.")
        # Continue execution even if data_directory is not set, but data loading will be skipped.
        # This allows the agent to potentially use web search fallback if implemented correctly.
        # exit() # Removed exit() here


    # Load all files directly from the folder (no subfolders assumed by load_multimodal_data_from_directory)
    # Only attempt to load data if data_directory is set
    if data_directory and os.path.isdir(data_directory):
        multimodal_data = load_multimodal_data_from_directory(data_directory)

        # Process multimodal data (extract video frames, audio segments)
        # Define processing parameters
        FRAMES_PER_SECOND = 1 # Number of frames to extract per second from videos
        AUDIO_SEGMENT_LENGTH_MS = 5000 # Length of audio segments in milliseconds
        processed_multimodal_data = process_multimodal_data(multimodal_data, FRAMES_PER_SECOND, AUDIO_SEGMENT_LENGTH_MS)


        # Generate embeddings for the processed data items
        print(f"\nGenerating embeddings for {len(processed_multimodal_data)} items...")
        # Initialize dictionaries to store processed data items and embeddings organized by modality type
        modality_data_items = {}
        modality_embeddings = {}

        if multimodal_embedding_model and processed_multimodal_data:
            for i, item in enumerate(processed_multimodal_data):
                item_id = item.get('id', f'item_{i}') # Use .get() for safety
                item_type = item.get('type') # Use .get() for safety

                if not item_type:
                    print(f"Warning: Item {item_id} has no type. Skipping embedding generation.")
                    continue

                try:
                    embedding = None
                    # Generate embedding based on item type
                    if item_type in ['text', 'pdf']: # Treat PDF content as text for embedding
                        if 'text' in item and item['text']:
                            embedding = multimodal_embedding_model.get_embedding(item['text'], 'text')

                    elif item_type in ['image', 'video_frame']: # Treat video frames as images for embedding
                         if 'image' in item and isinstance(item['image'], Image.Image):
                            embedding = multimodal_embedding_model.get_embedding(item['image'], 'image')

                    elif item_type == 'audio_segment':
                         # For audio segments, pass the file path to the embedding model's audio embedding method
                         if 'audio_path' in item and os.path.exists(item['audio_path']):
                              embedding = multimodal_embedding_model.get_embedding(item['audio_path'], 'audio')
                         else:
                             print(f"Skipping embedding for audio segment {item_id}: audio path is missing or invalid.")
                             embedding = None

                    # Store the generated embedding if successful
                    if embedding is not None:
                        # Ensure the embedding is a 1D numpy array of float32 type
                        if isinstance(embedding, torch.Tensor):
                             embedding = embedding.squeeze().cpu().numpy().astype('float32')
                        elif isinstance(embedding, np.ndarray):
                             embedding = embedding.squeeze().astype('float32')

                        # Validate the dimension based on modality
                        expected_dim = None
                        if item_type in ['text', 'pdf']: expected_dim = TEXT_FEATURE_DIM
                        elif item_type in ['image', 'video_frame']: expected_dim = IMAGE_FEATURE_DIM
                        elif item_type == 'audio_segment': expected_dim = AUDIO_FEATURE_DIM # OpenL3 outputs 512

                        if embedding.ndim == 1 and (expected_dim is None or embedding.shape[0] == expected_dim):
                            item['embedding'] = embedding # Add embedding to the data item
                            multimodal_data_with_embeddings.append(item) # Add to the list of items with embeddings

                            # Organize items and embeddings by modality type
                            if item_type not in modality_data_items:
                                modality_data_items[item_type] = []
                            modality_data_items[item_type].append(item)

                            if item_type not in modality_embeddings:
                                modality_embeddings[item_type] = []
                            modality_embeddings[item_type].append(embedding)

                        else:
                            print(f"Warning: Embedding for item {item_id} ({item_type}) has unexpected shape ({embedding.shape}) or dimension ({embedding.shape[0]} vs expected {expected_dim}). Skipping storage.")

                except Exception as e:
                    print(f"Error generating embedding for item {item_id} ({item_type}): {e}")
                    traceback.print_exc() # Print traceback for embedding errors


            print(f"\nFinished generating embeddings. Successfully generated {len(multimodal_data_with_embeddings)} embeddings.")
            print(f"Stored {len(multimodal_data_with_embeddings)} data items with embeddings.")
            print(f"Populated modality_data_items for modalities: {list(modality_data_items.keys())}")
            print(f"Organized embeddings for modalities: {list(modality_embeddings.keys())}")
        else:
            print("MultimodalEmbeddingModel is not initialized or processed_multimodal_data is empty. Cannot generate embeddings or build modality_embeddings.")


        # Build FAISS indexes for each modality
        modality_indexes = {}
        if modality_embeddings:
            print("\nBuilding FAISS indexes for each modality...")
            for modality, embeddings_list in modality_embeddings.items():
                if embeddings_list:
                    try:
                        # Convert list of embeddings to a NumPy array of float32
                        embeddings_np = np.array(embeddings_list).astype('float32')
                        embedding_dimension = embeddings_np.shape[-1]

                        print(f"Building FAISS index for {modality} with dimension {embedding_dimension}...")
                        # Use IndexFlatL2 for Euclidean distance
                        index = faiss.IndexFlatL2(embedding_dimension)
                        # Add embeddings to the index
                        index.add(embeddings_np)
                        # Store the index in the dictionary
                        modality_indexes[modality] = index
                        print(f"FAISS index built for {modality} with {index.ntotal} embeddings.")
                    except Exception as e:
                        print(f"An error occurred while building the FAISS index for {modality}: {e}")
                        traceback.print_exc() # Print traceback for FAISS errors
                else:
                    print(f"No embeddings available for modality: {modality}. Skipping index creation.")
        else:
            print("No modality embeddings available. Cannot build FAISS indexes.")

    else: # If data_directory is not set or not a directory, skip data loading, processing, embedding, and indexing
        print("\nSkipping data loading, processing, embedding, and indexing because MULTIMODAL_DATA_DIRECTORY is not set or not a valid directory.")
        # Ensure modality_indexes and modality_data_items are empty dictionaries in this case
        modality_indexes = {}
        modality_data_items = {}


    # Define a relevance threshold for retrieval (lower distance = more relevant)
    # This value might need tuning based on the embedding model and data.
    RELEVANCE_THRESHOLD = 40.0 # Example threshold


    # Instantiate the SimpleAIAgent with all necessary components
    print("\nInstantiating SimpleAIAgent...")
    # Pass the instantiated embedding model, tokenizer, processor, FAISS indexes,
    # and the organized data items to the agent.
    # Also pass the embedding dimensions.
    # Removed the check for modality_indexes and modality_data_items being non-empty
    if multimodal_embedding_model and tokenizer and processor:
        agent = SimpleAIAgent(
            name="MultimodalAgent",
            embedding_model=multimodal_embedding_model,
            tokenizer=tokenizer,
            processor=processor,
            modality_indexes=modality_indexes, # Pass the modality-specific indexes (can be empty)
            modality_data_items=modality_data_items, # Pass the modality-specific data items (can be empty)
            text_feature_dim=TEXT_FEATURE_DIM,
            image_feature_dim=IMAGE_FEATURE_DIM,
            audio_feature_dim=AUDIO_FEATURE_DIM
        )
        print(f"SimpleAIAgent '{agent.name}' instantiated successfully.")
    else:
        print("Cannot instantiate SimpleAIAgent: Required components (Embedding model, tokenizer, or processor) are missing or not initialized.")
        agent = None # Ensure agent is None if instantiation fails


    # Implement the interactive query loop
    # This is where the user interacts with the system.
    if agent: # Only run the loop if the agent was successfully instantiated
        print("\n" + "="*50)
        print("--- Multimodal RAG System Ready ---")
        print("Enter your queries to search the multimodal data, or ask about a location to see a map or travel route.")
        print("Type 'quit' to exit the query loop.")
        print("="*50 + "\n")


        while True:
            query = input("Enter your query: ")

            if query.lower() == 'quit':
                print("Exiting query loop.")
                break

            # Delegate the query handling to the agent's orchestration workflow
            # The agent will understand the intent, decide the action, execute it,
            # and generate a response.
            agent.orchestrate_workflow(query)

            print("-" * 50) # Separator between queries
    else:
        print("\nSystem initialization failed. Cannot run the query loop.")
        print("Please check the error messages above for details on what failed.")