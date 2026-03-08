from dataclasses import dataclass
from typing import List, Tuple

from vector_store import Chunk


@dataclass
class EvidenceAssessment:
    decision: str  # "PASS" or "FAIL"
    reason: str
    top_score: float
    supporting_chunks: List[Chunk]


def assess_evidence(results: List[Tuple[float, Chunk]], min_top_score: float = 0.65) -> EvidenceAssessment:
    """
    Minimal evidence gate:
    - PASS if top retrieval score >= threshold
    - FAIL otherwise
    """
    if not results:
        return EvidenceAssessment(
            decision="FAIL",
            reason="No retrieval results returned.",
            top_score=0.0,
            supporting_chunks=[],
        )

    top_score, _top_chunk = results[0]

    if top_score < min_top_score:
        return EvidenceAssessment(
            decision="FAIL",
            reason=f"Top evidence score {top_score:.4f} is below threshold {min_top_score:.2f}. Evidence too weak.",
            top_score=top_score,
            supporting_chunks=[ch for _, ch in results],
        )

    return EvidenceAssessment(
        decision="PASS",
        reason=f"Top evidence score {top_score:.4f} meets threshold {min_top_score:.2f}.",
        top_score=top_score,
        supporting_chunks=[ch for _, ch in results],
    )