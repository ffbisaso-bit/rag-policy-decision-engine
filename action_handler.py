# rag/action_handler.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


def _clean_snippet(text: str, max_chars: int = 220) -> str:
    """Single-line, word-boundary snippet for evidence display."""
    s = " ".join(text.replace("\n", " ").split()).strip()
    if len(s) <= max_chars:
        return s

    cut = s[:max_chars]
    # cut on last space to avoid mid-word truncation
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + "..."


@dataclass(frozen=True)
class EscalationReport:
    query: str
    decision: str
    reason: str
    confidence: float
    evidence: List[Tuple[float, Any]]  # (score, Chunk)
    next_step: str
    snippet_chars: int = 220
    top_n: int = 5

    def render(self) -> str:
        lines: List[str] = []
        lines.append("ESCALATION REPORT")
        lines.append(f"Query: {self.query}")
        lines.append(f"Decision: {self.decision}")
        lines.append(f"Reason: {self.reason}")
        lines.append(f"Confidence: {self.confidence:.3f}")
        lines.append("")
        lines.append("Evidence retrieved:")

        shown = self.evidence[: self.top_n]
        for score, ch in shown:
            snippet = _clean_snippet(getattr(ch, "text", ""), max_chars=self.snippet_chars)
            chunk_id = getattr(ch, "chunk_id", "unknown_chunk")
            lines.append(f"- {chunk_id} (score={score:.3f}): {snippet}")

        lines.append("")
        lines.append("Recommended next step:")
        lines.append(f"- {self.next_step}")
        return "\n".join(lines)


def handle_action(final_decision, query: str, results: List[Tuple[float, Any]]) -> tuple[str, str]:
    """
    final_decision: decision_engine.FinalDecision
    results: list of (score, Chunk) from vector_store.search
    Returns (user_message, report_text)
    """
    if final_decision is None:
        return _fallback_escalate(
            query=query,
            results=results,
            reason="Action handler received final_decision=None",
        )

    action = getattr(final_decision, "action", None)
    reason = getattr(final_decision, "reason", "") or "No reason provided."
    confidence = float(getattr(final_decision, "confidence", 0.0) or 0.0)

    if action not in {"ACT", "REFUSE", "ESCALATE"}:
        return _fallback_escalate(
            query=query,
            results=results,
            reason=f"Invalid action state from decision engine: {action}",
        )

    if action == "ACT":
        # We will generate the final answer in the next module.
        # For now: return an auditable record + a safe placeholder.
        user_message = "Approved to proceed. Evidence and confidence gates passed."
        record: List[str] = []
        record.append("DECISION RECORD")
        record.append(f"Query: {query}")
        record.append("Decision: ACT")
        record.append(f"Reason: {reason}")
        record.append(f"Confidence: {confidence:.3f}")
        record.append("")
        record.append("Evidence (top matches):")
        for score, ch in results[:3]:
            snippet = _clean_snippet(getattr(ch, "text", ""), max_chars=220)
            chunk_id = getattr(ch, "chunk_id", "unknown_chunk")
            record.append(f"- {chunk_id} (score={score:.3f}): {snippet}")
        record.append("")
        record.append("Next: generate user-facing answer from evidence.")
        return user_message, "\n".join(record)

    if action == "REFUSE":
        user_message = "I can’t safely answer this. The evidence is insufficient or too risky."
        report: List[str] = []
        report.append("REFUSAL RECORD")
        report.append(f"Query: {query}")
        report.append("Decision: REFUSE")
        report.append(f"Reason: {reason}")
        report.append(f"Confidence: {confidence:.3f}")
        return user_message, "\n".join(report)

    # ESCALATE
    user_message = "This request needs review. I’ve prepared an escalation report with the evidence."

    q = query.lower()
    action_like = any(k in q for k in ["approve", "deny", "authorize", "refund", "delete", "close account", "escalate"])

    next_step = (
        "A supervisor should review and explicitly approve/deny this refund."
        if action_like
        else "A reviewer should confirm the correct policy clause/version to cite, then re-run the request."
    )

    report_obj = EscalationReport(
        query=query,
        decision="ESCALATE",
        reason=reason,
        confidence=confidence,
        evidence=results,
        next_step=next_step,
        snippet_chars=340,
        top_n=5,
    )
    return user_message, report_obj.render()


def _fallback_escalate(query: str, results: List[Tuple[float, Any]], reason: str) -> tuple[str, str]:
    user_message = "I can’t safely proceed. I’ve escalated this for review."
    report_obj = EscalationReport(
        query=query,
        decision="ESCALATE",
        reason=reason,
        confidence=0.0,
        evidence=results,
        next_step="Fix the action handler wiring, then re-run the pipeline.",
        snippet_chars=220,
        top_n=5,
    )
    return user_message, report_obj.render()