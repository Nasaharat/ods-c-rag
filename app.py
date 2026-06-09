"""Streamlit UI for the ODS-C study assistant."""

import os

import streamlit as st

from llm_client import LLMClient
from document_store import DocumentStore, read_pdf
from rag_pipeline import RAGPipeline

CORPUS_FOLDER = "corpus"
INDEX_FILE = "corpus_index.npz"


def get_pipeline():
    """Build and cache the pipeline, loading the corpus on first run."""
    if "pipeline" not in st.session_state:
        try:
            llm = LLMClient()
        except RuntimeError as error:
            st.error(str(error))
            st.stop()
        store = DocumentStore(llm)
        if os.path.exists(INDEX_FILE):
            store.load_index(INDEX_FILE)
        else:
            store.load_corpus(CORPUS_FOLDER)
        st.session_state.pipeline = RAGPipeline(store, llm)
        st.session_state.history = {}
    return st.session_state.pipeline


def home():
    st.title("ODS-C Study Assistant")
    st.write(
        "Ask questions about cancer registry abstracting and get answers "
        "grounded in the loaded documents, with sources shown. Built for "
        "people studying for the Oncology Data Specialist (ODS-C) exam."
    )
    st.write(
        "Set a user name in the sidebar, then use the **Documents** page to "
        "add your own files and the **Chat** page to ask questions."
    )


def documents(pipeline, user):
    st.title("Documents")
    st.caption(f"Signed in as: {user}")
    info = pipeline.document_store.documents_info()
    shared = {n: m for n, m in info.items() if m["owner"] == "shared"}
    yours = {n: m for n, m in info.items() if m["owner"] == user}

    st.subheader("Shared documents")
    if shared:
        for name, meta in shared.items():
            st.write(f"- {name} - tag: {meta['tag']} "
                     f"({meta['chunks']} chunks)")
    else:
        st.caption("None loaded.")

    st.subheader("Your uploads")
    if yours:
        for name, meta in yours.items():
            st.write(f"- {name} - tag: {meta['tag']} "
                     f"({meta['chunks']} chunks)")
    else:
        st.caption("You have not uploaded anything yet.")

    st.subheader("Upload documents")
    tag = st.text_input("Topic tag for uploads", value="general")
    uploaded = st.file_uploader(
        "Add .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    for file in uploaded or []:
        try:
            if file.name.endswith(".pdf"):
                text = read_pdf(file)
            else:
                text = file.read().decode("utf-8", errors="ignore")
            count = pipeline.document_store.add_document(
                file.name, text, user, tag or "general"
            )
            st.success(f"Added {file.name} ({count} chunks).")
        except Exception as error:
            st.error(f"Could not load {file.name}: {error}")


def chat(pipeline, user):
    st.title("Ask a Question")
    if pipeline.document_store.is_empty():
        st.warning("Load or upload documents first.")
        return

    history = st.session_state.history.setdefault(user, [])
    tags = ["All"] + pipeline.document_store.available_tags()
    tag = st.selectbox("Filter by topic", tags)
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input("Your question")
        submitted = st.form_submit_button("Ask")
    if submitted and question:
        with st.spinner("Searching..."):
            try:
                answer, refs, elapsed = pipeline.answer(question, user, tag)
            except RuntimeError as error:
                st.error(str(error))
                return
        history.append((question, answer, refs, elapsed))

    for question, answer, refs, elapsed in reversed(history):
        st.markdown(f"**Q: {question}**")
        st.write(answer)
        with st.expander("References"):
            for src, chunk, score in refs:
                st.markdown(f"*{src}* (similarity {score:.2f})")
                st.caption(chunk)
        st.caption(f"Answered in {elapsed:.1f}s")
        st.divider()


def main():
    st.set_page_config(page_title="ODS-C Study Assistant")
    pipeline = get_pipeline()

    user = st.sidebar.text_input("User", value="guest")
    page = st.sidebar.radio("Navigate", ["Home", "Documents", "Chat"])
    asked = len(st.session_state.history.get(user, []))
    st.sidebar.metric("Your questions", asked)

    if page == "Home":
        home()
    elif page == "Documents":
        documents(pipeline, user)
    else:
        chat(pipeline, user)


if __name__ == "__main__":
    main()
