# rag/chunker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    start: int
    end: int
    text: str


def _snap_start_to_word_boundary(text: str, start: int) -> int:
    """Move start left to a whitespace boundary (unless start == 0)."""
    if start <= 0:
        return 0
    i = start
    # Walk left until we hit whitespace
    while i > 0 and not text[i - 1].isspace():
        i -= 1
    return i


def _snap_end_to_word_boundary(text: str, end: int) -> int:
    """Move end right to a whitespace boundary (unless end == len(text))."""
    n = len(text)
    if end >= n:
        return n
    i = end
    # Walk right until we hit whitespace
    while i < n and not text[i].isspace():
        i += 1
    return i


def chunk_text(text: str, chunk_size: int = 220, overlap: int = 40) -> List[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    n = len(text)
    chunks: List[Chunk] = []

    start = 0
    chunk_idx = 0

    while start < n:
        raw_end = min(start + chunk_size, n)

        # Snap boundaries for cleaner excerpts
        snapped_start = _snap_start_to_word_boundary(text, start)
        snapped_end = _snap_end_to_word_boundary(text, raw_end)

        # Safety: avoid infinite loops if snapping collapses range
        if snapped_end <= snapped_start:
            snapped_start = start
            snapped_end = raw_end

        chunk_text_str = text[snapped_start:snapped_end].strip()

        chunks.append(
            Chunk(
                chunk_id=f"chunk_{chunk_idx:03d}",
                start=snapped_start,
                end=snapped_end,
                text=chunk_text_str,
            )
        )
        chunk_idx += 1

        if raw_end >= n:
            break

        # Advance with overlap (based on raw positions to keep spacing stable)
        start = max(raw_end - overlap, 0)

    return chunks