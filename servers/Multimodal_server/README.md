# Multimodal RAG System with Gemini 1.5 Flash

## Description

This project implements a Multimodal Retrieval Augmented Generation (RAG) system using the Google Gemini 1.5 Flash model. It is designed to process and understand information from various data modalities, including text, images, videos, audio, and PDFs, stored in a local directory.

The core of the system is an AI agent that can:
- Load and process multimodal data.
- Generate embeddings for different data types using specialized models (CLIP for text/images, OpenL3 for audio).
- Build and search modality-specific vector indexes (FAISS) for efficient retrieval of relevant information.
- Understand user intent from natural language queries.
- Perform RAG by retrieving relevant information from the local data store and using it as context for the Gemini model to generate informed responses. Includes a web search fallback if no relevant local data is found.
- Integrate with external tools for specific tasks like displaying maps, calculating travel time/routes, and searching for images online.

The system provides an interactive command-line interface where users can ask questions about their data or request specific actions like showing a map or finding images.

## Features

- **Multimodal Data Processing:** Handles various file types including `.txt`, `.pdf`, `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.mp4`, `.avi`, `.mov`, `.mkv`, `.mp3`, `.wav`, `.aac`, `.flac`. Extracts text from documents, frames from videos, and segments from audio files.
- **Multimodal Embedding Generation:** Utilizes pre-trained models (CLIP and OpenL3) to create vector representations (embeddings) for text, image, and audio data.
- **FAISS Indexing:** Builds efficient vector indexes (FAISS) for each data modality to enable fast similarity search based on embeddings.
- **Retrieval Augmented Generation (RAG):** Retrieves relevant local data based on the user's query embedding and provides this information as context to the Gemini model for generating informed responses. Includes a web search fallback if no relevant local data is found.
- **Intent Recognition:** Uses spaCy and keyword matching to identify the user's intent (e.g., asking a question about data, requesting a map, asking for travel directions, searching for images).
- **External Tool Integration:**
    -   **Map Display:** Uses Folium and Nominatim to display a map for a given location.
    -   **Travel Time and Route Calculation:** Uses Nominatim and OSRM to calculate travel time and visualize a route between two locations on a map.
    -   **Image Search and Display:** Uses the Google Custom Search API to find images online and displays them using Pillow.
- **Interactive Agent:** Provides a command-line interface for users to interact with the AI agent.

## Setup

To set up and run the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    The project requires several libraries. Install them using pip. A `requirements.txt` file is recommended, but based on the code, you'll need:
    ```bash
    pip install numpy faiss-cpu transformers torch Pillow opencv-python pydub pdfminer.six folium geopy requests polyline spacy google-generativeai google-api-python-client python-dotenv openl3 soundfile
    ```
    *Note: `faiss-cpu` is used here for simplicity. For GPU support, install `faiss-gpu`.*

4.  **Download spaCy model:** The agent uses spaCy for intent recognition. Download the required model:
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Set up environment variables:**
    -   Create a `.env` file in the project's root directory.
    -   Obtain API keys for the necessary services:
        -   **Google Gemini API:** Follow the instructions on the Google AI for Developers website to get an API key. Add it to your `.env` file:
            ```dotenv
            NEW_GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
            ```
        -   **Google Custom Search API (Optional, for web image search):** Obtain a Google Cloud API key and a Programmable Search Engine ID configured for image search. Add them to your `.env` file:
            ```dotenv
            GOOGLE_CLOUD_API_KEY=YOUR_GOOGLE_CLOUD_API_KEY
            PROGRAMMABLE_SEARCH=YOUR_PROGRAMMABLE_SEARCH_ENGINE_ID
            ```
            *Note: If these keys are not set, the image search feature will not work, but the rest of the system will function.*
    -   **Data Directory:** Specify the path to the directory containing your multimodal data files in the `.env` file:
        ```dotenv
        MULTIMODAL_DATA_DIRECTORY=path/to/your/multimodal/data
        ```
    *Ensure your `.env` file is correctly loaded by your environment.*

6.  **Place your data files:** Put your text, image, video, audio, and PDF files into the directory specified by `MULTIMODAL_DATA_DIRECTORY`. The system will load and process all files directly within this folder (subfolders are not recursively searched by default).

## Usage

1.  **Run the main script:** Execute the `main.py` file from your terminal with the virtual environment activated:
    ```bash
    python main.py
    ```

2.  **Interact with the agent:** The system will load and process your data, build indexes, and then prompt you to enter queries.
    ```
    --- Multimodal RAG System Ready ---
    Enter your queries to search the multimodal data, or ask about a location to see a map or travel route.
    Type 'quit' to exit the query loop.
    ==================================================

    Enter your query:
    ```

3.  **Enter your queries:** Type your questions or commands and press Enter. The agent will process your request, perform retrieval (if necessary), use external tools (if the intent is recognized), and generate a response.

    **Examples of Queries:**

    -   **RAG Queries:**
        -   "What is mentioned in the document about the history of the city?"
        -   "Describe the image of the landscape."
        -   "What is the main topic discussed in the audio file?"
        -   "Tell me about the content of the video."
        -   "Summarize the key points in the PDF file."

    -   **Map Display Queries:**
        -   "Show me a map of Paris."
        -   "Where is Tokyo?"
        -   "Map of London"

    -   **Travel Time/Route Queries:**
        -   "How long does it take to get from New York to Los Angeles?"
        -   "Calculate the travel time from London to Paris."
        -   "Show me the route from Berlin to Rome."
        -   "Directions from San Francisco to Seattle."

    -   **Image Search Queries:**
        -   "Show me images of the Eiffel Tower."
        -   "Find pictures of ancient ruins."
        -   "Images of tropical beaches."

4.  **Exit:** Type `quit` and press Enter to exit the query loop.

## Dependencies

The project relies on the following key libraries and external services:

-   **Python Libraries:**
    -   `numpy`: Numerical operations.
    -   `faiss-cpu` (or `faiss-gpu`): For efficient similarity search (vector indexing).
    -   `transformers`: To load tokenizer and processor for embedding models (e.g., CLIP).
    -   `torch`: PyTorch framework, used by the CLIP model.
    -   `Pillow` (PIL): Image processing.
    -   `opencv-python` (cv2): Video processing.
    -   `pydub`: Audio processing.
    -   `pdfminer.six`: PDF text extraction.
    -   `folium`: Map visualization.
    -   `geopy`: Geocoding (converting location names to coordinates).
    -   `requests`: Making HTTP requests (for web search, OSRM).
    -   `polyline`: Encoding/decoding polyline geometries (for OSRM routes).
    -   `spacy`: Natural Language Processing, used for intent recognition and entity extraction.
    -   `google-generativeai`: Interface for the Google Gemini API.
    -   `google-api-python-client`: Client library for Google APIs (used for Custom Search).
    -   `python-dotenv`: Loading environment variables from a `.env` file.
    -   `openl3`: For generating audio embeddings.
    -   `soundfile`: Reading audio files for OpenL3.

-   **External Services:**
    -   **Google Gemini API:** For generating responses (LLM).
    -   **Open Source Routing Machine (OSRM) Demo Server:** For calculating travel time and routes.
    -   **Nominatim (OpenStreetMap):** For geocoding locations.
    -   **Google Custom Search API (Optional):** For performing web image searches.
