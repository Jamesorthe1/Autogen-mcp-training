import sys
import os
import asyncio
import subprocess
import time
import re
import inspect
from dotenv import load_dotenv

# --- Environment Setup ---

load_dotenv(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autogen_config.llm_config import get_llm_config
from autogen.agentchat import AssistantAgent

# --- Constants ---

APP_VENV_PYTHON = os.getenv("APP_VENV_PYTHON")
STREAMLIT_APP_PATH = os.getenv("STREAMLIT_APP_PATH")

MULTIMODAL_VENV_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", ".venv", "Scripts", "python.exe")
)

AGENTIC_RAG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "servers", "Multimodal_server", "main.py")
)

STREAMLIT_PROCESS = None


# --- Streamlit Summarizer Service ---

def run_streamlit_app():
    global STREAMLIT_PROCESS

    if STREAMLIT_PROCESS and STREAMLIT_PROCESS.poll() is None:
        print("✅ Streamlit summarizer already running.")
        return

    try:
        print("🚀 Launching Streamlit summarizer app as a subservice...")
        STREAMLIT_PROCESS = subprocess.Popen(
            [APP_VENV_PYTHON, "-m", "streamlit", "run", STREAMLIT_APP_PATH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print("🟢 Streamlit summarizer started in background. Visit http://localhost:8501")
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {e}")


# --- Agentic RAG Tool ---

def run_agentic_rag():
    print("🔎 Launching Agentic Multimodal RAG for location search...\n")
    try:
        subprocess.run([MULTIMODAL_VENV_PYTHON, AGENTIC_RAG_PATH], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Agentic_Rag.py exited with an error: {e}")
    except FileNotFoundError:
        print("❌ Could not find Agentic_Rag.py or Python interpreter. Check the paths.")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
    print("\n↩️ Returned from Agentic RAG.\n")
    time.sleep(1)


# --- Assistant Agent Setup ---

async def create_assistant_agent():
    llm_config = get_llm_config()
    agent = AssistantAgent(name="assistant", llm_config=llm_config)
    return agent


# --- Safe LLM Call Wrapper ---

async def ask_agent(agent, message: str):
    result = agent.generate_reply(messages=[{"role": "user", "content": message}])
    if inspect.isawaitable(result):
        response = await result
    else:
        response = result

    if isinstance(response, dict):
        return response.get("content", str(response)).strip()
    if isinstance(response, str):
        return response.strip()
    return str(response)


# --- Intent Classification ---

ROUTER_PROMPT = """
You are a smart task router in an AI assistant. Your job is to read the user's message and decide what action should be taken.

Choose ONLY ONE of the following labels:
1. summarize_legal  — if the user wants to summarize a legal document or legal text
2. run_multimodal_rag — if the user wants to search for locations, multimodal information, or pictures

Just respond with ONE label only: summarize_legal or run_multimodal_rag. No explanations.

User message: "{message}"
"""

def extract_intent_label(text):
    match = re.search(r"(summarize_legal|run_multimodal_rag)", text, re.IGNORECASE)
    return match.group(1).lower() if match else None


# --- Main Loop ---
def extract_intent_label(text):
    """
    Extracts one of the known intent labels from the LLM response.
    Ensures strict matching and trims irrelevant text.
    """
    cleaned = text.strip().lower()

    # Look for exact labels first
    if "summarize_legal" in cleaned:
        return "summarize_legal"
    elif "run_multimodal_rag" in cleaned:
        return "run_multimodal_rag"

    # Fallback — try regex
    match = re.search(r"(summarize_legal|run_multimodal_rag)", cleaned)
    return match.group(1) if match else None

async def main():
    assistant = await create_assistant_agent()

    while True:
        user_input = input("🤖 What would you like me to do? (type 'exit' to quit)\n> ")
        if user_input.lower() in ("exit", "quit"):
            break

        print("🧠 Asking assistant agent to classify your intent...")

        try:
            router_prompt = ROUTER_PROMPT.format(message=user_input)
            router_response = await ask_agent(assistant, router_prompt)
            intent_label = extract_intent_label(router_response)
            print(f"🔎 Detected intent: {intent_label}")
        except Exception as e:
            print(f"⚠️ Error while routing task: {e}")
            intent_label = None
            

        if intent_label == "summarize_legal":
            run_streamlit_app()
            print("📄 Legal document summarizer is now running at: http://localhost:8501")

        elif intent_label == "run_multimodal_rag":
            run_agentic_rag()

        else:
            print("⚠️ Could not determine a valid task. Please try rephrasing your request.")
            print(f"🧾 Raw router response: {router_response}")

    # Cleanup
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
