import os
import pdfplumber
import textwrap
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)

model = SentenceTransformer(os.getenv('MODEL'))

# ---------- UTILS ----------
def extract_text_from_pdf(filestream):
    text = ''
    with pdfplumber.open(filestream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def chunk_text(text, max_tokens=500):
    return textwrap.wrap(text, max_tokens)

# ---------- ROUTES ----------
@app.route("/")
def index():
    return "SERVER IS RUNNING"

@app.route("/upload", methods=['POST'])
def upload_pdf():

    file = request.files['file']
    
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    
    text = extract_text_from_pdf(file.stream)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    
    return jsonify({"success": True, "message": "PDF Text extracted", "text": text}), 200

if __name__ == "__main__":
    app.run(debug=True)