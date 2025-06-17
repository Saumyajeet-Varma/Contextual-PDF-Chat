import os
import pdfplumber
import textwrap
import sqlite3
import uuid
import faiss
import requests
import numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify
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

def load_all_chunks_and_embeddings():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT chunk, embedding FROM documents")
    rows = c.fetchall()
    chunks = []
    embeddings = []
    for chunk, emb_blob in rows:
        chunks.append(chunk)
        embedding = np.frombuffer(emb_blob, dtype="float32")
        embeddings.append(embedding)
    conn.close()
    return chunks, embeddings

# LLM Integrated
def get_LLM_response(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",  
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"Error from LLM: {e}"
    return prompt
# ---------- ROUTES ----------
@app.route("/")
def index():
    return "SERVER IS RUNNING"

@app.route("/upload", methods=['POST'])
def upload_pdf():

    file = request.files['file']
    
    if not file:
        return jsonify({"success": False, "message": "No file uploaded"}), 400
    
    text = extract_text_from_pdf(file.stream)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)

    store_chunks(chunks, embeddings)
    
    return jsonify({"success": True, "message": "PDF processed and stored in permanent memory", "text": text}), 200

@app.route("/ask", methods=['POST'])
def get_answer():

    question = request.json.get("question")

    if not question:
        return jsonify({"success": False, "message": "No question asked"}), 400
    
    chunks, embeddings = load_all_chunks_and_embeddings()

    if not chunks:
        return jsonify({"success": False, "message": "No documents available in the memory"}), 400
    
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    query_embedding = model.encode([question])
    distance, indices = index.search(np.array(query_embedding), k=3)
    relative_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]

    context = "\n".join(relative_chunks)
    answer = get_LLM_response(question, context)

    return jsonify({"success": True, "message": "LLM Response generated", "answer": answer}), 200

if __name__ == "__main__":
    app.run(debug=True)