# rag/vector_store.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

from embedder import embed_text


@dataclass
class Chunk:
    chunk_id: str
    start: int
    end: int
    text: str


@dataclass
class VectorItem:
    chunk: Chunk
    embedding: List[float]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    # Safe cosine similarity (no numpy)
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom else 0.0


def _tokenize(text: str) -> List[str]:
    """
    Minimal tokenizer: lowercase, keep alnum, split on whitespace.
    This is just for lexical reranking (cheap signal), not NLP perfection.
    """
    cleaned = []
    for ch in text.lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return [t for t in "".join(cleaned).split() if t]


def _lexical_overlap_score(query: str, doc: str) -> float:
    """
    Simple lexical signal: proportion of unique query tokens present in doc.
    Range: 0..1
    """
    q = set(_tokenize(query))
    if not q:
        return 0.0
    d = set(_tokenize(doc))
    return len(q & d) / len(q)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.items: List[VectorItem] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        for ch in chunks:
            emb = embed_text(ch.text)
            self.items.append(VectorItem(chunk=ch, embedding=emb))

    def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        initial_k: int = 10,
        alpha: float = 0.85,
    ) -> List[Tuple[float, Chunk]]:
        """
        Two-stage retrieval:
        1) Dense retrieval (cosine similarity on embeddings) to get initial_k candidates
        2) Lexical rerank on those candidates
        Final score = alpha*dense + (1-alpha)*lexical

        Returns: List[(final_score, Chunk)] sorted desc
        """
        if not self.items:
            return []

        # Ensure initial_k is valid
        initial_k = max(top_k, min(initial_k, len(self.items)))

        # Stage 1: dense retrieval
        q_emb = embed_text(query)
        dense_scored: List[Tuple[float, Chunk]] = []
        for it in self.items:
            dense_score = cosine_similarity(q_emb, it.embedding)
            dense_scored.append((dense_score, it.chunk))

        dense_scored.sort(key=lambda x: x[0], reverse=True)
        candidates = dense_scored[:initial_k]

        # Stage 2: lexical rerank (cheap signal)
        reranked: List[Tuple[float, Chunk]] = []
        for dense_score, ch in candidates:
            lex = _lexical_overlap_score(query, ch.text)
            final_score = (alpha * float(dense_score)) + ((1.0 - alpha) * float(lex))
            reranked.append((final_score, ch))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]