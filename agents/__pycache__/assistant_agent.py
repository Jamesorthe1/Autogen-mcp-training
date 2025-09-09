import sys
import os
import asyncio
from dotenv import load_dotenv
import aiohttp
import fitz  # PyMuPDF
import subprocess
import time

# Load environment variables
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen_config.llm_config import get_llm_config
from autogen.agentchat import AssistantAgent

# Constants
LEGAL_DOCS_DIR = os.getenv("LEGAL_DOCS_DIR")
if not LEGAL_DOCS_DIR or not os.path.isdir(LEGAL_DOCS_DIR):
    print("⚠️ LEGAL_DOCS_DIR is not set or does not exist.")
    sys.exit(1)

MULTIMODAL_VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", ".venv", "Scripts", "python.exe")
)
AGENTIC_RAG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", "Agentic_Rag.py")
)

# --- Utility Functions ---

def read_latest_document_text():
    """Returns the text of the most recently modified document in the legal folder."""
    files = [f for f in os.listdir(LEGAL_DOCS_DIR) if f.lower().endswith((".txt", ".pdf"))]
    if not files:
        return None, "⚠️ No legal documents found."

    # Sort files by modification time
    files.sort(key=lambda f: os.path.getmtime(os.path.join(LEGAL_DOCS_DIR, f)), reverse=True)
    latest_file = files[0]
    file_path = os.path.join(LEGAL_DOCS_DIR, latest_file)

    try:
        if latest_file.lower().endswith(".pdf"):
            text = ""
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    text += page.get_text()
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        return text, None
    except Exception as e:
        return None, f"⚠️ Error reading document: {e}"

async def summarize_legal_document(text: str) -> str:
    url = "http://127.0.0.1:8001/indian_legal_summarize"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, json={"text": text}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("summary", "No summary available.")
                else:
                    return f"⚠️ API error {resp.status}: {await resp.text()}"
        except Exception as e:
            return f"⚠️ Error calling summarization API: {e}"

async def create_assistant_agent():
    llm_config = get_llm_config()
    agent = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
    )
    agent.register_function({"summarize_legal_document": summarize_legal_document})
    return agent

def run_agentic_rag():
    """Launches the Agentic_Rag multimodal tool and waits until it exits."""
    print("🔎 Launching Agentic Multimodal RAG for location search...\n")
    try:
        subprocess.run([MULTIMODAL_VENV_PYTHON, AGENTIC_RAG_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Agentic_Rag.py exited with an error: {e}")
    except FileNotFoundError:
        print("❌ Could not find Agentic_Rag.py or Python interpreter. Check the paths.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
    print("\n↩️ Returned from Agentic RAG. You're back with the Assistant.\n")
    time.sleep(1)  # Optional: small delay for user experience

# --- Main loop ---

async def main():
    assistant = await create_assistant_agent()

    while True:
        user_input = input("🤖 What would you like me to do? (type 'exit' to quit)\n> ")
        if user_input.lower() in ("exit", "quit"):
            break

        elif "summarize" in user_input.lower():
            doc_text, error = read_latest_document_text()
            if error:
                print(error)
                continue

            print("🕐 Summarizing the latest document, please wait...")
            summary = await summarize_legal_document(doc_text)
            print(f"\n📝 Summary:\n{summary}\n")

        elif "search" in user_input.lower() and "location" in user_input.lower():
            run_agentic_rag()

        else:
            print("💬 Passing input to the assistant agent...")
            response = await assistant.aask(user_input)
            print("🤖 Assistant:", response)

# --- Entry point ---

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
