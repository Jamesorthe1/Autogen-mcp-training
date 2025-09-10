from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import fitz  # PyMuPDF
import os
from transformers import pipeline

app = FastAPI()

# Load summarization pipeline using HF token env var
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    use_auth_token=os.getenv("HF_TOKEN")
)

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

class Document(BaseModel):
    text: str

@app.post("/summarize_text")
async def summarize_text(doc: Document):
    summary = summarizer(doc.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

@app.post("/summarize_pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
