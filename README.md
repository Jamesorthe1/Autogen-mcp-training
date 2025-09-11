
---

```markdown
# 🤖 AI Assistant with Task Routing and Specialized Tools

A modular AI assistant that intelligently routes user requests to dedicated tools using Large Language Models (LLMs). Built for flexibility, it supports summarization of legal documents, multimodal search, and easy integration of new features.

---

## 📚 Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
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

```

.
├── .env
├── autogen\_requirements.txt
├── agents/
│   └── assistant\_agent.py
├── autogen\_config/
│   └── llm\_config.py
└── servers/
├── Indian\_Legal\_doc\_Summarizer/
│   ├── .venv/
│   └── app.py
└── Multimodal\_server/
├── .venv/
└── main.py

````

---

## ⚙️ Getting Started

### 🔧 Setup

#### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
````

#### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r autogen_requirements.txt
```

#### 4. Configure Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OLLAMA_API_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=

# --- Legal Document Summarizer ---
APP_VENV_PYTHON=/path/to/Indian_Legal_doc_Summarizer/.venv/bin/python
STREAMLIT_APP_PATH=/path/to/Indian_Legal_doc_Summarizer/app.py

# --- Multimodal Search Tool ---
# MULTIMODAL_VENV_PYTHON=/path/to/Multimodal_server/.venv/bin/python
# AGENTIC_RAG_PATH=/path/to/Multimodal_server/main.py
```

> 💡 *Paths are auto-detected by the assistant but can be overridden if needed.*

#### 5. Pull the Ollama Model

```bash
ollama pull llama3:instruct
```

---

### ▶️ Running the Assistant

#### 1. Activate the Virtual Environment

```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### 2. Start the Assistant

```bash
python agents/assistant_agent.py
```

The assistant will start and wait for user input.

---

### 🧰 Using the Tools

#### ✅ Legal Document Summarization

**Prompt:**

```
summarize a legal document
```

* This will launch the Streamlit summarizer.
* Access it in your browser at: [http://localhost:8501](http://localhost:8501)

---

#### ✅ Multimodal Location Search

**Prompt:**

```
search for locations
```

* This triggers the Agentic RAG module to process your query.

---

#### ❌ Exit the Assistant

To quit, type:

```bash
exit
```

or

```bash
quit
```

---

## 📌 Notes

* Ensure **Ollama** is installed and running for model inference.
* The assistant is **modular** — additional tools can be integrated seamlessly.
* Designed for experimentation and extension in multi-agent workflows.

---

```

---

✅ **Now you can copy this entire code block into a `README.md` file without needing to piece anything together.**

Let me know if you want this saved as a downloadable file too.
```
