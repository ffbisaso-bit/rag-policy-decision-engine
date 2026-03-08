# rag/arbitrator.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class ArbitrationResult:
    decision: str   # "OK" | "MUST_ESCALATE"
    reason: str


def _query_is_about_amount_or_approval(query: str) -> bool:
    q = query.lower()
    triggers = [
        "£", "$", "amount", "price", "cost", "above", "over", "exceed",
        "approve", "approval", "automatically", "auto", "refund above",
        "refund over", "financial", "payment", "chargeback"
    ]
    return any(t in q for t in triggers)


def _chunk_contains_escalation_clause(text: str) -> bool:
    t = text.lower()
    # Keep this small and explicit for the first product
    return (
        "must escalate" in t
        or "requires" in t and "review" in t
        or "human supervisor" in t
        or "manager approval" in t
        or "financial department review" in t
    )


def arbitrate(query: str, results: List[Tuple[float, Any]], min_rule_score: float = 0.60) -> ArbitrationResult:
    """
    Query-aware arbitration:
    - Only enforce escalation clauses if the user query is about approvals/amounts/high-stakes actions.
    - Require the clause to appear in a *highly relevant* chunk (score threshold), not a weak match.
    """
    if not results:
        return ArbitrationResult(decision="OK", reason="No retrieved evidence; nothing to arbitrate.")

    # If the query is not about amounts/approval, do NOT trigger escalation rules.
    if not _query_is_about_amount_or_approval(query):
        return ArbitrationResult(
            decision="OK",
            reason="Query not about approval/amount/risk action; ignoring unrelated escalation clauses in retrieval.",
        )

    # If query IS about approval/amount, enforce escalation rules only when strongly supported.
    for score, ch in results:
        text = getattr(ch, "text", "") or ""
        if score >= min_rule_score and _chunk_contains_escalation_clause(text):
            return ArbitrationResult(
                decision="MUST_ESCALATE",
                reason="High-relevance evidence contains an explicit escalation/review requirement.",
            )

    return ArbitrationResult(
        decision="OK",
        reason=f"No escalation clause found above relevance threshold {min_rule_score:.2f}.",
    )