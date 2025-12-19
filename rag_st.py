import streamlit as st
from app import run_rag_pipeline

st.set_page_config(page_title="Document Search", layout="wide")
st.title("📚 Document Search & Retrieval")

query = st.text_input("🔎 Enter your query")
top_k = st.slider("Number of documents", 1, 5, 3)

if st.button("Search") and query.strip():
    with st.spinner("Searching documents..."):
        answer, docs = run_rag_pipeline(query, top_k)

    st.subheader("🧠 Retrieved Content")
    st.write(answer)
