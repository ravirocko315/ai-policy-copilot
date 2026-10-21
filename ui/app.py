import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid_search import HybridSearcher
from src.generation.rag_pipeline import PolicyCopilot
from ui.presentation import clean_answer_for_display


@st.cache_resource
def load_copilot():
    docs = load_documents(settings.data_dir)
    chunks = chunk_documents(
        docs,
        max_tokens=settings.max_chunk_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )

    vector_store = VectorStore(dim=settings.embedding_dim)
    vector_store.build(chunks)

    bm25_store = BM25Store()
    bm25_store.build(chunks)

    searcher = HybridSearcher(vector_store, bm25_store)
    return PolicyCopilot(searcher)


st.set_page_config(page_title="AI Policy Copilot", page_icon="📄")

EXAMPLE_QUESTIONS = [
    "How many days of paid annual leave am I entitled to per year?",
    "What is the annual maximum coverage under the health insurance plan?",
    "How often must I change my password?",
]

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

with st.sidebar:
    st.header("About")
    st.write(
        "This is a demo RAG system over HR, Insurance, and Enterprise "
        "Security policy documents. It uses hybrid search (dense plus "
        "keyword), a reranker, and returns answers with citations."
    )
    st.caption(
        "Running in offline mode unless an OpenAI API key is set, so "
        "answers are shown as the most relevant policy excerpts."
    )

    st.header("Try an example")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q):
            st.session_state.question_input = q

st.title("📄 AI Policy Copilot")
st.write("Ask a question about HR, Insurance, or Security policies.")

copilot = load_copilot()

question = st.text_input("Your question:", key="question_input")

if question:
    with st.spinner("Searching policies..."):
        response = copilot.answer(question)

    st.subheader("Answer")
    st.write(clean_answer_for_display(response.answer))

    st.subheader("Sources")
    if not response.citations:
        st.write("No relevant policy sections were found for this question.")
    else:
        for citation in response.citations:
            label = f"{citation.marker} {citation.doc_title} - {citation.section_title} (score: {citation.relevance_score})"
            with st.expander(label):
                st.write(citation.snippet)
