import os
from dotenv import load_dotenv

load_dotenv()

print("Loaded HF Token:", os.getenv("HF_TOKEN"))


# HF_TOKEN=