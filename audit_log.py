# rag/audit_log.py

from __future__ import annotations
import hashlib
import uuid

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


def _fingerprint_supporting_chunks(assessment: Any) -> str:
    """
    Stable fingerprint for the evidence basis.
    Hashes the supporting chunk ids + text that your assessor already stores.
    """
    chunks = getattr(assessment, "supporting_chunks", None) or []
    payload = []
    for ch in chunks:
        cid = getattr(ch, "chunk_id", "")
        txt = getattr(ch, "text", "")
        payload.append(f"{cid}:{txt}")
    joined = "\n".join(payload).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()[:16]


def _safe_snippet(text: str, max_chars: int = 220) -> str:
    s = " ".join(text.replace("\n", " ").split()).strip()
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut + "..."

def write_audit_record(
    *,
    log_path: str,
    query: str,
    assessment: Any,
    gate: Any,
    arb: Any,
    final: Any,
    results: List[Tuple[float, Any]],
    user_answer: str | None = None,
    pipeline_config: Dict[str, Any] | None = None,
    doc_fingerprint: str | None = None,
    run_id: str | None = None,
) -> None:
    """
    Writes one JSON line per run. Replayable and easy to grep.
    """
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def pack(obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if is_dataclass(obj):
            return asdict(obj)
        out = {}
        for k in ("decision", "reason", "confidence", "action", "top_score", "supporting_chunks"):
            if hasattr(obj, k):
                out[k] = getattr(obj, k)
        return out

    evidence = []
    for score, ch in results:
        evidence.append(
            {
                "chunk_id": getattr(ch, "chunk_id", "unknown_chunk"),
                "score": float(score),
                "snippet": _safe_snippet(getattr(ch, "text", ""), max_chars=220),
            }
        )

    record = {
        "run_id": str(uuid.uuid4()),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "pipeline_config": pipeline_config,
        "doc_fingerprint": doc_fingerprint,
        "evidence_assessment": pack(assessment),
        "confidence_gate": pack(gate),
        "arbitration": pack(arb),
        "final_decision": pack(final),
        "evidence": evidence,
        "user_answer": user_answer,
    }

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")