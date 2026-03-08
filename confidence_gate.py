# rag/confidence_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple


@dataclass(frozen=True)
class GateResult:
    decision: str            # "ALLOW" | "ESCALATE" | "BLOCK"
    confidence: float
    reason: str


def confidence_gate(
    query: str,
    results: List[Tuple[float, Any]],
    assessment: Any,
    *,
    allow_threshold: float = 0.70,
    # If confidence is below allow_threshold but still >= escalate_floor → ESCALATE
    escalate_floor: float | None = None,
) -> GateResult:
    """
    Evidence-first gate.

    - If evidence assessment didn't pass → BLOCK.
    - Else confidence := top retrieval score
    - If confidence >= allow_threshold → ALLOW
    - Else if confidence >= escalate_floor → ESCALATE
    - Else → BLOCK

    This lets you tune informational flows (lower allow_threshold),
    while keeping a deliberate 'review band' via escalate_floor.
    """

    # 1) If evidence failed, we do not allow the system to proceed.
    if getattr(assessment, "decision", None) != "PASS":
        conf = _top_score(results)
        return GateResult(
            decision="BLOCK",
            confidence=conf,
            reason="Evidence assessment did not pass. Blocked.",
        )

    # 2) Confidence signal (simple v1): top similarity score
    conf = _top_score(results)

    # Default review band: 0.05 below allow threshold
    if escalate_floor is None:
        escalate_floor = max(0.0, allow_threshold - 0.05)

    # 3) Decide
    if conf >= allow_threshold:
        return GateResult(
            decision="ALLOW",
            confidence=conf,
            reason=f"Confidence {conf:.4f} >= {allow_threshold:.2f}. Allowed to answer.",
        )

    if conf >= escalate_floor:
        return GateResult(
            decision="ESCALATE",
            confidence=conf,
            reason=(
                f"Confidence {conf:.4f} in review band "
                f"[{escalate_floor:.2f}, {allow_threshold:.2f}). Escalate for review."
            ),
        )

    return GateResult(
        decision="BLOCK",
        confidence=conf,
        reason=f"Confidence {conf:.4f} < {escalate_floor:.2f}. Blocked.",
    )


def _top_score(results: List[Tuple[float, Any]]) -> float:
    if not results:
        return 0.0
    return float(results[0][0])