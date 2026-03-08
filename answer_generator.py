# rag/answer_generator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class UserAnswer:
    answer: str
    cited_chunk_ids: List[str]


def _clean(text: str) -> str:
    return " ".join(text.replace("\n", " ").split()).strip()


def _is_relevant(query: str, chunk_text: str) -> bool:
    """
    Very simple relevance filter for this first product.
    Later we’ll replace this with a judge model / claim-evidence mapping.
    """
    q = query.lower()
    t = chunk_text.lower()

    # For this demo: detect "refund within 30 days" questions
    if "refund" in q and ("30 day" in q or "within 30" in q):
        return ("refund" in t) and ("30 day" in t or "within 30" in t)

    # Fallback: accept top result only
    return True


def generate_user_answer(query: str, results: List[Tuple[float, Any]]) -> UserAnswer:
    """
    results: list of (score, Chunk) from vector_store.search
    Returns a clean, user-facing answer + chunk citations.
    """
    if not results:
        return UserAnswer(
            answer="I can’t answer safely because no evidence was retrieved.",
            cited_chunk_ids=[],
        )

    # Prefer relevant evidence; otherwise fall back to top chunk.
    relevant = []
    for score, ch in results:
        text = getattr(ch, "text", "")
        if _is_relevant(query, text):
            relevant.append((score, ch))

    use = relevant if relevant else [results[0]]

    top_score, top_chunk = use[0]
    top_text = _clean(getattr(top_chunk, "text", ""))

    # Hard-coded for this first policy demo: yes/no for “30 days” question.
    q = query.lower()
    if "refund" in q and ("30 day" in q or "within 30" in q):
        if "refund" in top_text.lower() and ("30 day" in top_text.lower() or "within 30" in top_text.lower()):
            cited = [getattr(top_chunk, "chunk_id", "unknown_chunk")]
            return UserAnswer(
                answer=f"Yes — refunds can be requested within 30 days of purchase. (Evidence: {', '.join(cited)})",
                cited_chunk_ids=cited,
            )

    # Generic fallback: cite the best available chunk only.
    cited = [getattr(top_chunk, "chunk_id", "unknown_chunk")]
    snippet = top_text[:180] + ("..." if len(top_text) > 180 else "")
    return UserAnswer(
        answer=f"Based on the retrieved policy, the key statement is: “{snippet}” (Evidence: {', '.join(cited)})",
        cited_chunk_ids=cited,
    )