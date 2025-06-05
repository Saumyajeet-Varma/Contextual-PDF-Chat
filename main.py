import os
import pdfplumber
import textwrap
import sqlite3
import uuid
import numpy as np
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)

model = SentenceTransformer(os.getenv('MODEL'))

db_path = os.getenv("DB_PATH")

# ---------- UTILS ----------
def init_db():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    chunk TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )''')
    conn.commit()
    conn.close()

init_db()

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

def store_chunks(chunks, embeddings):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for chunk, embedding in zip(chunks, embeddings):
        emb_bytes = np.array(embedding).astype('float32').tobytes()
        c.execute("INSERT INTO documents (id, chunk, embedding) VALUES (?, ?, ?)", (str(uuid.uuid4()), chunk, emb_bytes))
    conn.commit()
    conn.close()

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
    
    store_chunks(chunks, embeddings)
    
    return jsonify({"success": True, "message": "PDF processed and stored in permanent memory", "text": text}), 200

if __name__ == "__main__":
    app.run(debug=True)