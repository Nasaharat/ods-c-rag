"""Streamlit UI for the ODS-C study assistant."""

import streamlit as st

from llm_client import LLMClient
from document_store import DocumentStore, read_pdf
from rag_pipeline import RAGPipeline

CORPUS_FOLDER = "corpus"


def get_pipeline():
    """Build and cache the pipeline, loading the corpus on first run."""
    if "pipeline" not in st.session_state:
        try:
            llm = LLMClient()
        except RuntimeError as error:
            st.error(str(error))
            st.stop()
        store = DocumentStore(llm)
        store.load_corpus(CORPUS_FOLDER)
        st.session_state.pipeline = RAGPipeline(store, llm)
        st.session_state.history = []
    return st.session_state.pipeline


def home():
    st.title("ODS-C Study Assistant")
    st.write(
        "Ask questions about cancer registry abstracting and get answers "
        "grounded in the loaded documents, with sources shown. Built for "
        "people studying for the Oncology Data Specialist (ODS-C) exam."
    )
    st.write(
        "Use the **Documents** page to see or add reference files, and the "
        "**Chat** page to ask questions like "
        "*\"What does the behavior code describe?\"*"
    )


def documents(pipeline):
    st.title("Documents")
    sources = pipeline.document_store.loaded_sources()
    if sources:
        for name, count in sources.items():
            st.write(f"- {name} ({count} chunks)")
    else:
        st.warning("No documents loaded.")

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
            count = pipeline.document_store.add_document(file.name, text)
            st.success(f"Added {file.name} ({count} chunks).")
        except Exception as error:
            st.error(f"Could not load {file.name}: {error}")


def chat(pipeline):
    st.title("Ask a Question")
    if pipeline.document_store.is_empty():
        st.warning("Load or upload documents first.")
        return
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Searching..."):
            try:
                answer, refs, elapsed = pipeline.answer(question)
            except RuntimeError as error:
                st.error(str(error))
                return
        st.session_state.history.append((question, answer, refs, elapsed))
    for question, answer, refs, elapsed in reversed(st.session_state.history):
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
    page = st.sidebar.radio("Navigate", ["Home", "Documents", "Chat"])
    st.sidebar.metric("Questions asked", pipeline.query_count)
    if page == "Home":
        home()
    elif page == "Documents":
        documents(pipeline)
    else:
        chat(pipeline)


if __name__ == "__main__":
    main()
