AI Assistant with Legal Summarization & Multimodal RAG Search
=============================================================

This project provides a command-line based AI assistant that combines
LLM-driven conversations, legal document summarization via API, and
multimodal location-based search using an external RAG tool.

-------------------------------------------------------------
Project Structure
-------------------------------------------------------------

Main Script:
- scripts/assistant_main.py

Other Components:
- servers/Multimodal_server/Agentic_Rag.py     # Location-based RAG tool
- autogen_config/llm_config.py                 # LLM configuration
- .env                                         # Environment variables file

-------------------------------------------------------------
Features
-------------------------------------------------------------

1. LLM-Based Assistant
   - Uses AutoGen and OpenAI (or compatible) models
   - Responds to general user queries

2. Legal Document Summarization
   - Sends text to a local API: http://127.0.0.1:8001/indian_legal_summarize
   - Returns summarized content (currently text only)

3. Multimodal Location-Based Search
   - Launches 'Agentic_Rag.py' tool in a separate subprocess
   - Tool must be preconfigured and available in its virtual environment

-------------------------------------------------------------
Requirements
-------------------------------------------------------------

- Python 3.8 or newer
- Virtual environment for Agentic_Rag.py located at:
  servers/Multimodal_server/.venv/

- Local summarization API running at:
  http://127.0.0.1:8001/indian_legal_summarize

- Install dependencies:
  pip install -r requirements.txt

-Install and run an Ollama server(for ILDS)
 ollama serve
 
- Add a .env file at the root level with required keys, such as:
  OLLAMA_API_KEY=your-api-key (optional)

-------------------------------------------------------------
How to Run
-------------------------------------------------------------

1. Navigate to the 'scripts' directory.

2. Run the assistant script:
   python assistant_main.py

3. Enter commands in the prompt:
   - "summarize ..." to use summarization (currently text input only)
   - "search location ..." to launch the RAG tool
   - "exit" or "quit" to exit

-------------------------------------------------------------
Notes
-------------------------------------------------------------

- Make sure all paths, api keys and environments are correctly configured.
- The RAG tool returns control after execution completes.
- You may need to adjust paths for Linux/macOS systems.

-------------------------------------------------------------
Future Improvements
-------------------------------------------------------------

- Add error handling for LLM and API failures
- Create a simple web or GUI front-end

-------------------------------------------------------------
License
-------------------------------------------------------------

This project is provided under the MIT License.