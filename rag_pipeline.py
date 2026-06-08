"""Retrieval-augmented answering with simple guardrails."""

import re
import time

SYSTEM_PROMPT = (
    "You are a study assistant for the Oncology Data Specialist (ODS-C) "
    "certification. Answer using ONLY the context below, drawn from cancer "
    "registry reference material. Use a clear, encouraging study tone. If the "
    "answer is not in the context, say you cannot find it in the loaded "
    "documents instead of guessing.\n\nContext:\n{context}"
)

# Questions containing any of these are refused (e.g., injection attempts).
BLOCKED_TERMS = [
    "ignore previous instructions",
    "ignore the above",
    "disregard your instructions",
    "reveal your system prompt",
]

# Patterns redacted from retrieved text before it is shown or sent.
SENSITIVE_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
]


class RAGPipeline:
    """Retrieves relevant chunks and asks the LLM a grounded question."""

    def __init__(self, document_store, llm_client, top_k=4):
        self.document_store = document_store
        self.llm_client = llm_client
        self.top_k = top_k
        self.query_count = 0

    def answer(self, question, owner=None, tag=None):
        """Answer a question. Returns (answer, references, elapsed_seconds)."""
        start = time.time()
        if self._is_blocked(question):
            return (
                "That request can't be answered here. Please ask a study "
                "question about the loaded documents.",
                [],
                time.time() - start,
            )
        results = self.document_store.search(question, self.top_k, owner, tag)
        if not results:
            return ("No matching documents are loaded.", [],
                    time.time() - start)
        # Redact sensitive data before showing or sending the chunks.
        results = [(src, self._mask(chunk), score)
                   for src, chunk, score in results]
        context = "\n\n".join(
            f"[Source {i}: {src}]\n{chunk}"
            for i, (src, chunk, _) in enumerate(results, 1)
        )
        answer = self.llm_client.chat(
            SYSTEM_PROMPT.format(context=context), question
        )
        self.query_count += 1
        return answer, results, time.time() - start

    def _is_blocked(self, question):
        """Return True if the question matches a blocked pattern."""
        lowered = question.lower()
        return any(term in lowered for term in BLOCKED_TERMS)

    def _mask(self, text):
        """Replace anything that looks like sensitive data with [redacted]."""
        for pattern in SENSITIVE_PATTERNS:
            text = re.sub(pattern, "[redacted]", text)
        return text
