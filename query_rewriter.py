# rag/query_rewriter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RewriteResult:
    original: str
    primary: str
    alternates: List[str]
    reason: str


def rewrite_query(query: str) -> RewriteResult:
    """
    Simple, deterministic query rewriting (no LLM).
    Purpose: improve vector recall by generating a cleaner primary query + a few alternates.

    We do NOT invent facts. We only:
    - normalize phrasing
    - keep core entities/constraints
    - produce a few semantically-close variants
    """
    q = " ".join(query.strip().split())
    q_low = q.lower()

    # Basic normalization patterns for common policy questions
    # Keep it conservative: rephrase, don't expand with new details.
    primary = q

    # Normalize can/allowed phrasing → policy wording
    replacements = [
        ("can we", "is it allowed to"),
        ("can i", "is it allowed to"),
        ("can customers", "are customers allowed to"),
        ("automatically", "without human approval"),
    ]
    for a, b in replacements:
        if a in q_low:
            primary = _replace_case_insensitive(primary, a, b)

    # Add policy keywords if missing (helps retrieval on short queries)
    # Only add generic words, not facts.
    if "policy" not in primary.lower():
        primary = f"{primary} (policy)"

    # Alternates: keep short, high-signal
    alternates: List[str] = []

    # Variant 1: remove fluff parentheses but keep meaning
    alternates.append(primary.replace("(policy)", "").strip())

    # Variant 2: turn question into a statement
    alternates.append(_question_to_statement(primary))

    # Variant 3: explicitly ask for the relevant section
    alternates.append(f"Find the relevant policy section for: {query}")

    # De-dupe while preserving order
    seen = set()
    cleaned: List[str] = []
    for item in [primary] + alternates:
        item2 = " ".join(item.split()).strip()
        if item2 and item2 not in seen:
            seen.add(item2)
            cleaned.append(item2)

    # cleaned[0] is primary, rest are alternates
    primary_final = cleaned[0]
    alternates_final = cleaned[1:]

    reason = (
        "Generated a conservative primary rewrite + a few alternates to improve retrieval recall "
        "without adding new facts."
    )

    return RewriteResult(
        original=query,
        primary=primary_final,
        alternates=alternates_final,
        reason=reason,
    )


def _replace_case_insensitive(text: str, needle: str, repl: str) -> str:
    """Replace first occurrence, case-insensitive, preserving rest of string."""
    idx = text.lower().find(needle.lower())
    if idx == -1:
        return text
    return text[:idx] + repl + text[idx + len(needle) :]


def _question_to_statement(text: str) -> str:
    t = text.strip()
    if t.endswith("?"):
        t = t[:-1].strip()
    # crude but effective: "Is it allowed to X" -> "Allowed to X"
    low = t.lower()
    if low.startswith("is it allowed to "):
        return "Allowed to " + t[len("is it allowed to ") :]
    if low.startswith("are customers allowed to "):
        return "Customers are allowed to " + t[len("are customers allowed to ") :]
    return t