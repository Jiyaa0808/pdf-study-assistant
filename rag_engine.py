import fitz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_pdf_text(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def split_into_chunks(text):
    words = text.split()
    return [" ".join(words[i:i+500]) for i in range(0, len(words), 450)]

def create_vectorstore(chunks):
    if not chunks:
        return {"chunks": [], "embeddings": np.array([])}
    embeddings = model.encode(chunks)
    return {"chunks": chunks, "embeddings": embeddings}

def search(query, db):
    # Guard: empty or missing embeddings
    if db is None or len(db["embeddings"]) == 0:
        return []
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, db["embeddings"])[0]
    top_indices = np.argsort(scores)[-3:][::-1]
    return [db["chunks"][i] for i in top_indices]