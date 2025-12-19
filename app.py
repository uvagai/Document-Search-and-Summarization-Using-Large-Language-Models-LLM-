import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------- LOAD DOCUMENTS ----------
def load_documents(data_dir="data"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    texts = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)

    return texts


# ---------- BUILD VECTOR INDEX ----------
def build_faiss_index(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))

    return index, model


# ---------- RAG PIPELINE ----------
def run_rag_pipeline(query, top_k=3):
    texts = load_documents("data")
    index, model = build_faiss_index(texts)

    query_embedding = model.encode([query]).astype("float32")
    _, indices = index.search(query_embedding, top_k)

    results = [texts[i] for i in indices[0]]
    answer = "\n\n".join(results)

    return answer, results
