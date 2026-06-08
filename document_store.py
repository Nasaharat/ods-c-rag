"""Stores document chunks and their embeddings, and searches over them."""

import os
import re

import numpy as np


class DocumentStore:
    """Ingests documents, chunks and embeds them, and runs similarity search."""

    def __init__(self, llm_client, chunk_size=150, chunk_overlap=30):
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.sources = []
        self.embeddings = None

    def _chunk(self, text):
        """Clean text and split it into overlapping word windows."""
        words = re.sub(r"\s+", " ", text).strip().split()
        step = self.chunk_size - self.chunk_overlap
        return [" ".join(words[i:i + self.chunk_size])
                for i in range(0, len(words), step)]

    def add_document(self, name, text):
        """Chunk, embed, and store one document. Returns the chunk count."""
        chunks = self._chunk(text)
        if not chunks:
            return 0
        vectors = np.array(self.llm_client.embed(chunks), dtype=float)
        self.chunks.extend(chunks)
        self.sources.extend([name] * len(chunks))
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        return len(chunks)

    def load_corpus(self, folder):
        """Load every .txt and .md file in a folder. Returns name -> count."""
        loaded = {}
        if not os.path.isdir(folder):
            return loaded
        for name in sorted(os.listdir(folder)):
            if name.endswith((".txt", ".md")):
                with open(os.path.join(folder, name), encoding="utf-8") as f:
                    loaded[name] = self.add_document(name, f.read())
        return loaded

    def loaded_sources(self):
        """Return a dict of source name -> number of stored chunks."""
        counts = {}
        for source in self.sources:
            counts[source] = counts.get(source, 0) + 1
        return counts

    def is_empty(self):
        return not self.chunks

    def search(self, query, k=4):
        """Return the top-k chunks as (source, chunk, score) tuples."""
        if self.is_empty():
            return []
        q = np.array(self.llm_client.embed([query])[0], dtype=float)
        q = q / (np.linalg.norm(q) + 1e-10)
        rows = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        )
        scores = rows @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.sources[i], self.chunks[i], float(scores[i])) for i in top]
