# ODS-C Study Assistant

A retrieval-augmented (RAG) study companion for the Oncology Data Specialist
(ODS-C) exam. Ask a question and get an answer grounded in the loaded cancer
registry documents, with the source passages shown.

**Live demo:** https://ods-c-rag-8txzhndvfsr77tun8gcunq.streamlit.app/

## How it works

1. Documents (.txt, .md, .pdf) are cleaned and split into overlapping chunks.
2. Each chunk is embedded with the OpenAI embeddings API (stored in memory).
3. A question is embedded and matched to chunks by cosine similarity.
4. The top chunks become context for a grounded chat-model prompt.

## Features

- Three-page UI: Home, Documents, and Chat.
- Grounded answers that cite the source chunks (with similarity scores).
- PDF, text, and markdown ingestion.
- Per-user document sets: set a user name, and your uploads stay yours while
  everyone shares the starter corpus.
- Topic tags with a retrieval filter, so you can narrow questions to a subject.
- Guardrails: prompt-injection questions are refused, and sensitive data
  (SSNs, emails, phone numbers) is redacted from retrieved text.
- Simple analytics: a question counter and per-answer response time.

## Structure

```
app.py             Streamlit UI (Home, Documents, Chat)
rag_pipeline.py    RAGPipeline: retrieval, prompt, guardrails, LLM call
document_store.py  DocumentStore: chunking, embeddings, metadata, search
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

Put `.txt`, `.md`, or `.pdf` files in `corpus/` to load on launch, or upload
them on the Documents page during a session.
