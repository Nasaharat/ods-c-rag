"""Retrieval-augmented answering: retrieve chunks, then ask the LLM."""

import time

SYSTEM_PROMPT = (
    "You are a study assistant for the Oncology Data Specialist (ODS-C) "
    "certification. Answer using ONLY the context below, drawn from cancer "
    "registry reference material. Use a clear, encouraging study tone. If the "
    "answer is not in the context, say you cannot find it in the loaded "
    "documents instead of guessing.\n\nContext:\n{context}"
)


class RAGPipeline:
    """Retrieves relevant chunks and asks the LLM a grounded question."""

    def __init__(self, document_store, llm_client, top_k=4):
        self.document_store = document_store
        self.llm_client = llm_client
        self.top_k = top_k
        self.query_count = 0

    def answer(self, question):
        """Answer a question. Returns (answer, references, elapsed_seconds)."""
        start = time.time()
        results = self.document_store.search(question, self.top_k)
        if not results:
            return ("No documents are loaded yet.", [], time.time() - start)
        context = "\n\n".join(
            f"[Source {i}: {src}]\n{chunk}"
            for i, (src, chunk, _) in enumerate(results, 1)
        )
        answer = self.llm_client.chat(SYSTEM_PROMPT.format(context=context), question)
        self.query_count += 1
        return answer, results, time.time() - start
