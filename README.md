# 🤖 AI Assistant with Task Routing and Specialized Tools

A modular AI assistant that intelligently routes user requests to dedicated tools using Large Language Models (LLMs). Built for flexibility, it supports summarization of legal documents, multimodal search, and easy integration of new features.

---

## 📚 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Setup](#setup)  
- [Running the Assistant](#running-the-assistant)  
- [Using the Tools](#using-the-tools)  
- [Notes](#notes)  

---

## 🧠 Overview

This project implements an AI assistant capable of handling diverse user requests by leveraging LLMs for task classification and tool routing. It includes specialized applications like:

- A **legal document summarizer**
- A **multimodal search engine**
- An **extensible architecture** for additional tools

---

## 🚀 Features

- **Task Routing**  
  Automatically classifies user intent and selects the appropriate tool.

- **Legal Document Summarizer**  
  A Streamlit app for summarizing Indian legal documents.

- **Multimodal Location Search**  
  Uses Agentic Retrieval-Augmented Generation (RAG) to process multimodal queries.

- **Extensible Architecture**  
  Built to integrate new tools easily with minimal changes.

---

## 🗂️ Project Structure

```plaintext
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

⚙️ Getting Started
🔧 Setup

Clone the Repository

git clone <repository_url>
cd <repository_directory>


Create and Activate a Virtual Environment

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install Dependencies

pip install -r autogen_requirements.txt


Configure Environment Variables

Create a .env file in the root directory and add the following:

OLLAMA_API_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=

# --- Legal Document Summarizer ---
APP_VENV_PYTHON=/path/to/Indian_Legal_doc_Summarizer/.venv/bin/python
STREAMLIT_APP_PATH=/path/to/Indian_Legal_doc_Summarizer/app.py

# --- Multimodal Search Tool ---
# MULTIMODAL_VENV_PYTHON=/path/to/Multimodal_server/.venv/bin/python
# AGENTIC_RAG_PATH=/path/to/Multimodal_server/main.py


💡 Paths are auto-detected by the assistant, but you can override them.

Pull the Ollama Model

ollama pull llama3:instruct

▶️ Running the Assistant

Activate the Virtual Environment

source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Start the Assistant

python agents/assistant_agent.py


The assistant will start and wait for user input.

🧰 Using the Tools
✅ Legal Document Summarization

Say something like:

summarize a legal document


This will launch the Streamlit app.

Visit: http://localhost:8501
 in your browser.

✅ Multimodal Location Search

Say something like:

search for locations


This triggers the Agentic RAG module to process your query.

❌ Exit the Assistant

To quit, simply type:

exit


or

quit
