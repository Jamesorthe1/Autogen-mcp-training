import os
from dotenv import load_dotenv
import streamlit as st
import pdfplumber
from transformers import pipeline
import asyncio
import base64

# async loop handling
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables from .env file
load_dotenv()

# Fetch token
hf_token = os.getenv("HF_TOKEN")

# CACHE THE SUMMARIZER MODEL
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn", token=hf_token)

summarizer = load_summarizer()

# EXTRACT TEXT FROM PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text if text else "[ERROR] No text found in the PDF. Try another file."

# HTML & CSS STYLING
st.markdown(
    """
    <style>
        body {background-color: #f4f4f4;}
        .main {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1);}
        h1 {color: #0078FF; text-align: center;}
        .stButton>button {background-color: #0078FF; color: white; font-size: 18px; padding: 10px; border-radius: 5px;}
        .stTextInput>div>div>input {border-radius: 5px;}
    </style>
    """,
    unsafe_allow_html=True
)

# STREAMLIT UI
st.markdown("<h1>📜 Indian Legal Document Summarizer</h1>", unsafe_allow_html=True)
st.write("Upload a legal document in PDF format to generate a concise summary.")

uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["pdf"], help="Upload a PDF file for summarization")

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    
    if text.startswith("[ERROR]"):
        st.error(text)
    else:
        st.success("Text extracted successfully! Generating summary...")
        with st.spinner("Summarizing..."):
            summary = summarizer(text[:1024], max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        
        st.markdown("### 📝 Summary:")
        st.info(summary)

        # Download Button
        summary_bytes = summary.encode('utf-8')
        b64 = base64.b64encode(summary_bytes).decode()
        href = f'<a href = "data:file/txt;base64,{b64}" download = "Summary.txt">📥 Download Summary</a>'
        st.markdown(href, unsafe_allow_html=True)


