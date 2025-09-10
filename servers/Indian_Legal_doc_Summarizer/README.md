# Indian Legal Document Summarizer
![image](https://github.com/user-attachments/assets/545bd0c6-bcec-4d2a-99a0-22666219eba2)

## Overview
The **Indian Legal Document Summarizer** is a web-based application built using Streamlit that extracts and summarizes legal documents in PDF format. The tool leverages the **Facebook BART Large CNN** model from Hugging Face for text summarization, ensuring concise and meaningful summaries of lengthy legal texts.

## Features
- **PDF Upload**: Users can upload legal documents in PDF format.
- **Text Extraction**: Extracts text from PDFs using `pdfplumber`.
- **AI-Powered Summarization**: Utilizes the Hugging Face `facebook/bart-large-cnn` model to generate concise summaries.
- **Downloadable Summary**: Users can download the generated summary as a text file.
- **Interactive UI**: A professional and user-friendly interface powered by Streamlit.

## Installation
### Prerequisites
Ensure you have Python installed (preferably 3.8 or above). Then, install the required dependencies.

```sh
pip install -r requirements.txt
```

### Required Dependencies
Below are the major dependencies used in this project:
- `streamlit`
- `pdfplumber`
- `transformers`
- `python-dotenv`

Alternatively, you can install them individually using:

```sh
pip install streamlit pdfplumber transformers python-dotenv
```

## Usage
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/Indian-Legal-Document-Summarizer.git
   cd Indian-Legal-Document-Summarizer
   ```
2. Create a `.env` file and add your Hugging Face API token:
   ```sh
   HF_TOKEN=your_huggingface_token
   ```
3. Ensure `.env` is added to `.gitignore` to prevent accidental commits:
   ```sh
   echo .env >> .gitignore
   ```
4. Run the application:
   ```sh
   streamlit run app.py
   ```
5. Upload a PDF file and receive a concise summary.
6. Download the generated summary for future reference.

## Project Structure
```
Indian-Legal-Document-Summarizer/
│── app.py               # Main application file
│── requirements.txt     # Required dependencies
│── .gitignore           # Ignore sensitive files
│── .env.example         # Sample environment variables file
│── README.md            # Project documentation
```

## Notes
- Ensure your `.env` file contains a valid Hugging Face token.
- The model processes only the first 1024 characters of the document for summarization.
- The summarization process may take some time depending on the document size.

## License
This project is intended for educational and research purposes only. Feel free to use, modify, and share it responsibly.

## Author
Developed by **Praveen Roy**. Feel free to reach out for improvements or collaborations!
