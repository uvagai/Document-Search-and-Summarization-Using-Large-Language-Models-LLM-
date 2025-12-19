import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- LOAD & PREPROCESS DOCUMENTS ----------
def load_documents(data_dir="data"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    texts = []

    if not os.path.exists(data_path):
        return []

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                content = f.read().lower().strip()   # preprocessing
                if content:
                    texts.append(content)

    return texts


# ---------- EXTRACTIVE SUMMARIZATION ----------
def summarize(texts, max_sentences=5):
    summary_sentences = []
    for text in texts:
        sentences = text.split(".")
        summary_sentences.extend(sentences[:max_sentences])
    return ". ".join(summary_sentences)


# ---------- RAG PIPELINE ----------
def run_rag_pipeline(query, top_k=3, summary_len=5):
    texts = load_documents("data")

    # No documents available
    if not texts:
        return None, None

    model = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = model.encode(texts)
    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_docs = [texts[i] for i in top_indices]

    summary = summarize(retrieved_docs, max_sentences=summary_len)

    return summary, retrieved_docs
