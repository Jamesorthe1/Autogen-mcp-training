import os

def get_llm_config():
    return {
        "config_list": [
            {
                "api_type": "ollama",
                "model": "llama2",  # or "mistral" or any Ollama-supported model
                "base_url": os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434"),
                # API key usually not needed for local Ollama server; set if required
                #"api_key": os.getenv("OLLAMA_API_KEY", None),
            }
        ]
    }
