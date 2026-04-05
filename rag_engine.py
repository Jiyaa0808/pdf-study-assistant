import fitz
import chromadb
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()

def get_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def split_into_chunks(text):
    words = text.split()
    return [" ".join(words[i:i+500]) for i in range(0, len(words), 450)]

def create_vectorstore(chunks):
    try:
        client.delete_collection("pdf")
    except:
        pass
    db = client.create_collection("pdf")
    db.add(
        documents=chunks,
        embeddings=model.encode(chunks).tolist(),
        ids=[str(i) for i in range(len(chunks))]
    )
    return db

def search(query, db):
    results = db.query(query_embeddings=model.encode([query]).tolist(), n_results=3)
    return results["documents"][0]