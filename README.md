AI Assistant with Task Routing and Specialized Tools
This project implements a versatile AI assistant designed to handle various user requests by routing them to specialized tools. It leverages the power of Large Language Models (LLMs) to understand user intent and execute tasks efficiently.

Features
Task Routing: The core of the assistant is an LLM that classifies user intent and directs the request to the most appropriate tool for a given task.

Legal Document Summarizer: A dedicated Streamlit application for summarizing legal documents.

Multimodal Location Search: A separate tool that uses Agentic Retrieval-Augmented Generation (RAG) to find and process multimodal information related to locations.

Extensible Architecture: The system is built to be easily expanded with new tools and functionalities as needed.

Project Structure
The project has a modular structure to keep different functionalities separate and organized:

.
├── .env
├── autogen_requirements.txt
├── agents/
│   └── assistant_agent.py
├── autogen_config/
│   └── llm_config.py
└── servers/
    ├── Indian_Legal_doc_Summarizer/
    │   ├── .venv/
    │   └── app.py
    └── Multimodal_server/
        ├── .venv/
        └── main.py



Setup
Follow these steps to get the project running on your local machine.

1. Clone the Repository
git clone <repository_url>
cd <repository_directory>



2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

python -m venv .venv
source .venv/bin/activate  # Use `.venv\Scripts\activate` on Windows



3. Install Dependencies
Install the required Python packages from the autogen_requirements.txt file.

pip install -r autogen_requirements.txt



4. Configure Environment Variables
Create a .env file in the root directory of the project. This file will hold all the necessary configuration variables, including API URLs and file paths.

OLLAMA_API_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=

# --- Constants for Indian Legal Document Summarizer (ILDS) ---
APP_VENV_PYTHON=/path/to/your/Indian_Legal_doc_Summarizer/.venv/bin/python
STREAMLIT_APP_PATH=/path/to/your/Indian_Legal_doc_Summarizer/app.py

# --- Constants for Multimodal RAG Tool ---
# Note: The assistant_agent.py script automatically calculates these paths
# based on its own location. Adjust if your directory structure is different.
# MULTIMODAL_VENV_PYTHON=/path/to/your/Multimodal_server/.venv/bin/python
# AGENTIC_RAG_PATH=/path/to/your/Multimodal_server/main.py



5. Pull Ollama Model
Ensure you have Ollama installed and running. Then, pull the required language model (default is llama3:instruct).

ollama pull llama3:instruct



Running the AI Assistant
Once all the dependencies are installed and the environment variables are set up, you can start the assistant.

Activate the Virtual Environment
If you've closed your terminal, remember to reactivate the virtual environment.

source .venv/bin/activate  # Use `.venv\Scripts\activate` on Windows



Start the Assistant
Run the main Python script for the assistant agent.

python agents/assistant_agent.py



The assistant will start and prompt you for input.

Using the Tools
The assistant will automatically route your requests to the correct tool.

To Summarize a Legal Document: Just type a request like "summarize a legal document." The assistant will launch the Streamlit app, which you can then access in your web browser at http://localhost:8501.

To Perform a Multimodal Search: Type a request such as "search for locations." The assistant will execute the Agentic RAG script to handle your query.

To exit the assistant, simply type exit or quit when prompted
