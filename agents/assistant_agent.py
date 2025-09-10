import sys
import os
import asyncio
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen_config.llm_config import get_llm_config
from autogen.agentchat import AssistantAgent

# --- Constants ---

APP_VENV_PYTHON = r"C:\Users\spnes\OneDrive\Documents\industrial training to do\autogen_mcp_project\servers\Indian_Legal_doc_Summarizer\.venv\Scripts\python.exe"
STREAMLIT_APP_PATH = r"C:\Users\spnes\OneDrive\Documents\industrial training to do\autogen_mcp_project\servers\Indian_Legal_doc_Summarizer\app.py"

MULTIMODAL_VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", ".venv", "Scripts", "python.exe")
)

AGENTIC_RAG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", "Agentic_Rag.py")
)

STREAMLIT_PROCESS = None  # Will hold the Streamlit subprocess


# --- Streamlit Summarizer Service ---

def run_streamlit_app():
    """Launches the Streamlit app as a background subprocess if not already running."""
    global STREAMLIT_PROCESS

    if STREAMLIT_PROCESS is not None and STREAMLIT_PROCESS.poll() is None:
        print("✅ Streamlit summarizer already running.")
        return

    try:
        print("🚀 Launching Streamlit summarizer app as a subservice...")
        STREAMLIT_PROCESS = subprocess.Popen(
            [APP_VENV_PYTHON, "-m", "streamlit", "run", STREAMLIT_APP_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE  # Windows only — remove if on Linux/macOS
        )
        print("🟢 Streamlit summarizer started in background. Visit http://localhost:8501")
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")


# --- Assistant Agent Creation ---

async def create_assistant_agent():
    llm_config = get_llm_config()
    agent = AssistantAgent(
        name="assistant",
        llm_config=llm_config,
    )
    return agent


# --- Agentic RAG Tool ---

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
    time.sleep(1)


# --- Main Application Loop ---

async def main():
    # Create assistant agent
    assistant = await create_assistant_agent()

    while True:
        user_input = input("🤖 What would you like me to do? (type 'exit' to quit)\n> ")
        if user_input.lower() in ("exit", "quit"):
            break

        # Keywords that should trigger the legal document summarizer
        elif any(keyword in user_input.lower() for keyword in ["summarize", "legal", "legal document", "legal summary", "legal text"]):
            run_streamlit_app()
            print("📄 Legal document summarizer is now running at: http://localhost:8501")

        elif "search" in user_input.lower() and "location" in user_input.lower():
            run_agentic_rag()

        else:
            print("💬 Passing input to the assistant agent...")
            try:
                response = await assistant.aask(user_input)
                print("🤖 Assistant:", response)
            except Exception as e:
                print(f"⚠️ Assistant error: {e}")

    # Clean up Streamlit process on exit
    if STREAMLIT_PROCESS:
        print("🛑 Terminating Streamlit summarizer...")
        STREAMLIT_PROCESS.terminate()
        try:
            STREAMLIT_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            STREAMLIT_PROCESS.kill()
        print("✅ Streamlit summarizer terminated.")


# --- Entry Point ---

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
