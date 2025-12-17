import os
from transformers import pipeline

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


# ---------- LOAD DOCUMENTS ----------
def load_documents(data_dir="data"):
    documents = []

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found: {data_dir}")

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            path = os.path.join(data_dir, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append(Document(page_content=text))

    if not documents:
        raise ValueError("No text files found in data folder")

    return documents


# ---------- BUILD VECTOR STORE ----------
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------- BUILD LOCAL LLM ----------
def build_llm():
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        tokenizer="facebook/bart-large-cnn"
    )

    llm = HuggingFacePipeline(pipeline=summarizer)
    return llm


# ---------- BUILD RAG CHAIN ----------
def build_rag_chain(vectorstore, top_k=3):
    llm = build_llm()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True
    )
    return qa_chain


# ---------- FULL RAG PIPELINE ----------
def run_rag_pipeline(query, top_k=3):
    documents = load_documents("data")
    vectorstore = build_vectorstore(documents)
    rag_chain = build_rag_chain(vectorstore, top_k)

    response = rag_chain(query)

    sources = response["source_documents"]
    summary = response["result"]

    return summary, sources
