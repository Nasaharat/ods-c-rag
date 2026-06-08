# ODS-C Study Assistant

A retrieval-augmented (RAG) study companion for the Oncology Data Specialist
(ODS-C) exam. Ask a question and get an answer grounded in the loaded cancer
registry documents, with the source passages shown.

**Live demo:** _add your deployed URL here_

## How it works

1. Documents are cleaned and split into overlapping chunks.
2. Each chunk is embedded with the OpenAI embeddings API (stored in memory).
3. A question is embedded and matched to chunks by cosine similarity.
4. The top chunks become context for a grounded chat-model prompt.

## Structure

```
app.py             Streamlit UI (Home, Documents, Chat)
rag_pipeline.py    RAGPipeline: retrieval + prompt + LLM call
document_store.py  DocumentStore: chunking, embeddings, search
llm_client.py      LLMClient: OpenAI wrapper
corpus/            starter documents loaded on launch
```

## Run locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"   # Windows: set OPENAI_API_KEY=...
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Create an app at share.streamlit.io pointing to `app.py`.
3. In Settings > Secrets, add: `OPENAI_API_KEY = "your-key"`.
4. Deploy. Dependencies install from `requirements.txt`.

## Adding documents

Put `.txt` or `.md` files in `corpus/`, or upload them on the Documents page.
