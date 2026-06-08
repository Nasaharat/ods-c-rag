"""Stores document chunks with metadata and searches over them."""

import os
import re

import numpy as np


class DocumentStore:
    """Ingests, chunks, and embeds documents, then searches them."""

    def __init__(self, llm_client, chunk_size=150, chunk_overlap=30):
        self.llm_client = llm_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
        self.sources = []
        self.owners = []
        self.tags = []
        self.embeddings = None

    def _chunk(self, text):
        """Clean text and split it into overlapping word windows."""
        words = re.sub(r"\s+", " ", text).strip().split()
        step = self.chunk_size - self.chunk_overlap
        return [" ".join(words[i:i + self.chunk_size])
                for i in range(0, len(words), step)]

    def add_document(self, name, text, owner="shared", tag="General"):
        """Chunk, embed, and store a document with owner and tag metadata."""
        chunks = self._chunk(text)
        if not chunks:
            return 0
        vectors = np.array(self.llm_client.embed(chunks), dtype=float)
        self.chunks.extend(chunks)
        self.sources.extend([name] * len(chunks))
        self.owners.extend([owner] * len(chunks))
        self.tags.extend([tag] * len(chunks))
        if self.embeddings is None:
            self.embeddings = vectors
        else:
            self.embeddings = np.vstack([self.embeddings, vectors])
        return len(chunks)

    def load_corpus(self, folder):
        """Load every .txt, .md, and .pdf file in a folder as shared docs."""
        loaded = {}
        if not os.path.isdir(folder):
            return loaded
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if name.endswith((".txt", ".md")):
                with open(path, encoding="utf-8") as file:
                    text = file.read()
            elif name.endswith(".pdf"):
                with open(path, "rb") as file:
                    text = read_pdf(file)
            else:
                continue
            tag = os.path.splitext(name)[0]
            loaded[name] = self.add_document(name, text, "shared", tag)
        return loaded

    def documents_info(self):
        """Return {source: {owner, tag, chunks}} for each document."""
        info = {}
        for source, owner, tag in zip(self.sources, self.owners, self.tags):
            if source not in info:
                info[source] = {"owner": owner, "tag": tag, "chunks": 0}
            info[source]["chunks"] += 1
        return info

    def available_tags(self):
        """Return the sorted set of tags currently stored."""
        return sorted(set(self.tags))

    def is_empty(self):
        return not self.chunks

    def search(self, query, k=4, owner=None, tag=None):
        """Return top-k (source, chunk, score), filtered by owner and tag."""
        if self.is_empty():
            return []
        allowed = [
            i for i in range(len(self.chunks))
            if (owner is None or self.owners[i] in ("shared", owner))
            and (tag in (None, "All") or self.tags[i] == tag)
        ]
        if not allowed:
            return []
        q = np.array(self.llm_client.embed([query])[0], dtype=float)
        q = q / (np.linalg.norm(q) + 1e-10)
        sub = self.embeddings[allowed]
        sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-10)
        scores = sub @ q
        order = np.argsort(scores)[::-1][:k]
        return [
            (self.sources[allowed[j]], self.chunks[allowed[j]],
             float(scores[j]))
            for j in order
        ]


def read_pdf(file):
    """Extract text from a PDF given a path or a file-like object."""
    from pypdf import PdfReader
    text = ""
    for page in PdfReader(file).pages:
        text += (page.extract_text() or "") + " "
    return text
