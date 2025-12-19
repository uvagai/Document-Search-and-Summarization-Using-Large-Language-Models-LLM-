import streamlit as st
import textwrap
from app import run_rag_pipeline


st.set_page_config(page_title="Document Search", layout="wide")
st.title("📚 Document Search (RAG)")

query = st.text_input("🔎 Enter your query")

top_k = st.slider("Number of documents", 1, 5, 3)

if st.button("Search") and query.strip():
    with st.spinner("Searching documents..."):
        answer, docs = run_rag_pipeline(query, top_k)

    st.subheader("🧠 Retrieved Answer")
    st.write(answer)

    st.subheader("📄 Source Documents")
    for i, doc in enumerate(docs, 1):
        st.markdown(f"**Document {i}**")
        st.write(textwrap.shorten(doc.page_content, 400))
