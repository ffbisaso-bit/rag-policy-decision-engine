# rag/decision_engine.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinalDecision:
    action: str          # "ACT" | "REFUSE" | "ESCALATE"
    confidence: float
    reason: str


def decide(assessment, gate, arb) -> FinalDecision:
    """
    Priority order:
    1) Evidence must pass or we refuse.
    2) Confidence gate can block or escalate.
    3) Arbitration can force escalation.
    4) Otherwise act.
    """

    # Evidence failure => refuse
    if getattr(assessment, "decision", None) != "PASS":
        return FinalDecision(
            action="REFUSE",
            confidence=float(getattr(assessment, "top_score", 0.0) or 0.0),
            reason=f"Evidence assessment failed: {getattr(assessment, 'reason', '')}".strip(),
        )

    # Confidence gate decisions
    gate_decision = getattr(gate, "decision", None)
    conf = float(getattr(gate, "confidence", 0.0) or 0.0)
    gate_reason = (getattr(gate, "reason", "") or "").strip()

    if gate_decision == "BLOCK":
        return FinalDecision(action="REFUSE", confidence=conf, reason=f"Confidence gate blocked: {gate_reason}")

    if gate_decision == "ESCALATE":
        return FinalDecision(action="ESCALATE", confidence=conf, reason=f"Confidence gate escalation: {gate_reason}")

    # Arbitration can force escalation even if gate allows
    arb_decision = getattr(arb, "decision", None)
    arb_reason = (getattr(arb, "reason", "") or "").strip()

    if arb_decision in {"MUST_ESCALATE", "ESCALATE"}:
        return FinalDecision(action="ESCALATE", confidence=conf, reason=f"Arbitration triggered escalation: {arb_reason}")

    # Otherwise act
    return FinalDecision(action="ACT", confidence=conf, reason="Evidence and confidence gates passed; no arbitration conflicts.")