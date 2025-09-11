import torch
import numpy as np
import openl3
import soundfile as sf
from transformers import CLIPProcessor, CLIPModel
import os # Import os for checking file existence
from PIL import Image # Import the Image class

class MultimodalEmbeddingModel:
    """
    A class for generating multimodal embeddings using CLIP (for text and images)
    and OpenL3 (for audio).
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32"):
        """
        Initializes the CLIP and OpenL3 models for multimodal embedding generation.

        Args:
            clip_model_name (str): The name of the pre-trained CLIP model to load.
        """
        # Load pre-trained CLIP model and processor
        print(f"Loading CLIP model: {clip_model_name}")
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        print("CLIP model and processor loaded.")

        # Load OpenL3 model for audio embeddings
        print("Loading OpenL3 audio embedding model...")
        # Use a try-except block for OpenL3 loading as it can sometimes fail
        try:
            self.openl3_model = openl3.models.load_audio_embedding_model(
                input_repr="mel128", content_type="env", embedding_size=512
            )
            print("OpenL3 model loaded.")
        except Exception as e:
            print(f"Error loading OpenL3 model: {e}")
            self.openl3_model = None # Ensure model is None if loading fails


        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        print(f"Using device: {self.device}")


    def get_embedding(self, data, data_type):
        """
        Generates an embedding for a given data item based on its type.

        Args:
            data: The content (text string, PIL Image, or file path for audio).
            data_type (str): The type of data ('text', 'image', or 'audio').

        Returns:
            np.ndarray: A NumPy array representing the embedding, or None if embedding generation fails.

        Raises:
            ValueError: If an unsupported data type is provided.
        """
        if data_type == 'text':
            # Process text data using CLIP
            if not isinstance(data, str):
                print(f"Warning: Expected string data for text embedding, but got {type(data)}. Skipping.")
                return None
            try:
                inputs = self.clip_processor(text=data, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device) # Added truncation and max_length
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**inputs)
                # Ensure output is 1D
                return text_features.squeeze().cpu().numpy()
            except Exception as e:
                print(f"Error generating text embedding: {e}")
                return None

        elif data_type == 'image':
            # Process image data using CLIP
            # Handle single PIL image or a list of images (e.g., from a video)
            if not (isinstance(data, Image.Image) or (isinstance(data, list) and all(isinstance(img, Image.Image) for img in data))):
                 print(f"Warning: Expected PIL Image or list of PIL Images for image embedding, but got {type(data)}. Skipping.")
                 return None
            try:
                inputs = self.clip_processor(images=data, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**inputs)
                # Ensure output is 1D (for single image) or 2D (for list of images)
                return image_features.squeeze().cpu().numpy()
            except Exception as e:
                print(f"Error generating image embedding: {e}")
                return None


        elif data_type == 'audio':
            # Process audio data using OpenL3
            if self.openl3_model is None:
                print("Warning: OpenL3 model not loaded. Cannot generate audio embedding.")
                return None

            # Assuming 'data' is the audio path for OpenL3
            if isinstance(data, str) and os.path.exists(data):
                 try:
                      audio, sr = sf.read(data)
                      # OpenL3 takes the entire audio signal, or 1-second segments
                      # We process segments in the processing step, here we get the embedding for the whole audio file
                      # A more refined approach would be to get embeddings per segment and average/pool them.
                      # For now, let's use the whole audio if data is a path.
                      emb, ts = openl3.get_audio_embedding(
                          audio, sr, model=self.openl3_model, verbose=False
                      )
                      # If OpenL3 returns multiple embeddings (e.g., for segments), average them
                      if emb.ndim > 1:
                          return np.mean(emb, axis=0)
                      return emb.squeeze() # Ensure the output is 1D
                 except Exception as openl3_e:
                      print(f"Error generating OpenL3 embedding for {data}: {openl3_e}")
                      return None # Return None if embedding fails


            else:
                 print(f"Warning: Invalid data type or path for audio embedding: {type(data)}. Expected a valid file path.")
                 return None # Return None for invalid audio data


        else:
            raise ValueError(f"Unsupported data type: {data_type}")