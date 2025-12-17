import streamlit as st
import textwrap

from app import run_rag_pipeline


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Document Search & Summarization",
    layout="wide"
)

st.title("📚 Document Search & Summarization (RAG)")
st.write(
    "LangChain-based Retrieval-Augmented Generation using local models "
    "(SentenceTransformers + BART) with ROUGE evaluation."
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of documents to retrieve", 1, 10, 3)

# ---------- QUERY INPUT ----------
query = st.text_input(
    "🔎 Enter your query",
    placeholder="e.g. Explain machine learning in simple terms"
)

# ---------- RUN PIPELINE ----------
if st.button("Search & Summarize") and query.strip():
    with st.spinner("Running RAG pipeline..."):
        summary, sources = run_rag_pipeline(query, top_k)


    # ---------- RETRIEVED DOCUMENTS ----------
    st.subheader("📄 Retrieved Documents")
    for i, doc in enumerate(sources, 1):
        st.markdown(f"**Document {i}**")
        st.write(textwrap.shorten(doc.page_content, width=400, placeholder="..."))
        st.markdown("---")

    # ---------- SUMMARY ----------
    st.subheader("📝 Generated Summary")
    st.write(summary)

    # ---------- ROUGE ----------
   