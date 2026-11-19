"""Markdown-aware chunking.

Strategy: split each document on headings, prefix every chunk with its
document title and section heading (so a chunk that says "30 days" still
carries "Soft delete / Business"), then window long sections by characters
with overlap. Small, deterministic, and easy to reason about in an eval.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    section: str
    metadata: dict = field(default_factory=dict)


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # prefer to break on a sentence or line boundary inside the window
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in out if w]


def chunk_markdown(text: str, source: str, size: int = 700, overlap: int = 100) -> list[Chunk]:
    title, section, buf = source, "", []
    sections: list[tuple[str, str]] = []

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((section, body))

    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            flush()
            buf = []
            if len(m.group(1)) == 1:
                title = m.group(2).strip()
                section = ""
            else:
                section = m.group(2).strip()
        else:
            buf.append(line)
    flush()

    chunks: list[Chunk] = []
    for sec, body in sections:
        header = f"{title} > {sec}" if sec else title
        for i, piece in enumerate(_windows(body, size, overlap)):
            content = f"[{header}]\n{piece}"
            cid = hashlib.sha1(f"{source}|{sec}|{i}|{piece}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, content, source, sec, {"source": source, "section": sec, "part": i}))
    return chunks


def load_corpus(corpus_dir: Path, size: int = 700, overlap: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path.read_text(encoding="utf-8"), path.name, size, overlap))
    return chunks
