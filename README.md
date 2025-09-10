%%writefile README.txt
# Multimodal Tourism RAG with MCP Client

This Python script implements a multimodal Retrieval-Augmented Generation (RAG) system focused on tourism, integrating various data types (text, images, video frames, audio segments) and connecting to a Model Context Protocol (MCP) client for legal document summarization.

## Purpose

The primary purpose of this script is to demonstrate a RAG system capable of processing and retrieving information from diverse data sources. It utilizes multimodal embeddings and FAISS indexing for efficient similarity search. Additionally, it includes functionalities for location-based searches, displaying maps, calculating travel times, and summarizing legal documents via an external MCP client.

## Features

- **Multimodal Data Loading and Processing:** Handles text, PDF, image, video, and audio files from a specified directory.
- **Multimodal Embedding Generation:** Uses a combination of CLIP (for text and images) and OpenL3 (for audio) models to create embeddings. Video frames are processed as images, and audio files are segmented and processed.
- **FAISS Indexing:** Builds modality-specific FAISS indexes for efficient similarity search of embeddings.
- **Multimodal Retrieval:** Supports searching across different modalities using a query embedding and FAISS indexes. Includes a relevance threshold for filtering results and a web search fallback.
- **Gemini Integration:** Utilizes the Gemini API to generate responses based on the user's query and the retrieved multimodal context.
- **Location-Based Functionality:** Extracts locations from queries, performs image searches for locations, and displays interactive maps using Folium. Calculates travel time and route using OSRM.
- **Legal Document Summarization:** Connects to an external Model Context Protocol (MCP) client to summarize legal documents (PDF and TXT) from a designated directory.

## Flow

The script follows these main steps:

1.  **Initialization:** Imports necessary libraries and configures API keys from environment variables.
2.  **MultimodalEmbeddingModel Instantiation:** Initializes the CLIP and OpenL3 models for embedding generation.
3.  **Data Loading:** Loads data from the directory specified by the `MULTIMODAL_DATA_DIRECTORY` environment variable, identifying file types by extension.
4.  **Data Processing:** Processes loaded data, extracting text from PDFs, frames from videos, and segments from audio files.
5.  **Embedding Generation:** Generates embeddings for processed data items using the initialized multimodal embedding model.
6.  **FAISS Indexing:** Builds FAISS indexes for each data modality with generated embeddings.
7.  **Interactive Query Loop:** Enters a loop to handle user queries:
    *   Prompts the user to enter a query and select a search type ('location', 'multimodal', or 'legal').
    *   **Legal Document Summarization:** If 'legal' is selected, MCP client is called and new tab is opened using streamlit to host a site for uploading legal documents for summarization.
    *   **Location Search:** If 'location' is selected, extracts a location from the query, performs an image search, displays images, and generates and displays a map for the location. Optionally calculates and displays travel time and route from a source location.
    *   **Multimodal Search:** If 'multimodal' is selected, generates a query embedding, searches the FAISS indexes for relevant local data, includes a web search fallback if needed, and generates a response using the Gemini API based on the query and retrieved context.
    *   Exits the loop if the user types 'quit'.

## Code Documentation

This section provides a more detailed look at the key functions and classes within the script.

### `summarize_via_mcp(text)`

This function sends text to an external Model Context Protocol (MCP) client for summarization. It makes an HTTP POST request to a specified endpoint and returns the summary received from the client.

### `load_multimodal_data_from_directory(data_directory)`

This function iterates through all files in a given directory, identifies their type based on file extension, and loads the data accordingly. It supports text, PDF, image, video, and audio files.

### `MultimodalEmbeddingModel` Class

This class is responsible for generating multimodal embeddings. It utilizes:
- **CLIP:** For generating embeddings from text and images (including video frames).
- **OpenL3:** For generating embeddings from audio data (processing audio files as segments).

The `get_embedding(data, data_type)` method within this class takes data and its type as input and returns the corresponding embedding as a NumPy array.

### `retrieve_relevant_data_with_fallback(query_text, multimodal_embedding_model, modality_indexes, modality_data_items, k_per_modality, relevance_threshold)`

This is the core retrieval function. It performs the following steps:
1. Generates an embedding for the user's query using the `MultimodalEmbeddingModel`.
2. Queries the modality-specific FAISS indexes to find the nearest neighbors (most similar items) for each modality.
3. Filters the local results based on a `relevance_threshold`.
4. If no relevant local results are found, it performs a web search as a fallback mechanism using the Google Custom Search API.
5. Returns a combined list of relevant items from local indexes and web search, sorted by relevance (distance for local, order from web search).

### `generate_response_with_gemini(query, relevant_items)`

This function uses the Gemini API to generate a natural language response to the user's query, incorporating the information from the `relevant_items` retrieved by the `retrieve_relevant_data_with_fallback` function. It constructs a prompt including the query and the retrieved context and sends it to a suitable Gemini model (preferably a flash model).

### `display_map_for_location(location_name)`

This function uses the `geopy` library to get the coordinates of a given location name and then generates an interactive map centered at that location using the `folium` library. The map is saved to a temporary HTML file and opened in the default web browser.

### `display_travel_time_and_route(source_location, destination_location)`

This function calculates the estimated travel time and displays the driving route between two locations on a Folium map. It uses `geopy` to get coordinates and the OSRM (Open Source Routing Machine) API to get routing information. The route is displayed as a polyline on the map, which is saved to a temporary HTML file and opened in a browser.

### `search_images(query, num)`

This function performs an image search using the Google Custom Search API based on a given query. It returns a list of image URLs.

### `display_images(image_urls)`

This function takes a list of image URLs, downloads each image, and attempts to display it using Pillow's `show()` method, which typically opens the image in a default image viewer.

### `extract_location_from_query(query)`

This function uses the `spaCy` library with a pre-trained English model (`en_core_web_sm`) to perform Named Entity Recognition (NER) on the user's query and extract potential location names (entities labeled as GPE, LOC, ORG, or FAC).

### Main Execution Block (`if __name__ == "__main__":`)

This block orchestrates the entire process:
1. Initializes variables and the `MultimodalEmbeddingModel`.
2. Loads, processes, and generates embeddings for the multimodal data from the specified directory.
3. Builds FAISS indexes for each modality.
4. Enters an interactive loop to accept user queries.
5. Based on user input, it either:
    - Summarizes a legal document using the MCP client.
    - Performs location-based tasks (image search, map, travel time).
    - Performs a multimodal RAG search using the FAISS indexes and web search fallback, then generates a response with Gemini.
    - Exits the loop if the user types 'quit'.

## How to Run

1.  Ensure you have Python installed.
2.  Install the required libraries using the provided `requirements.txt` file:
