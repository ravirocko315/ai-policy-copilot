from dataclasses import dataclass
from typing import List, Tuple
from src.ingestion.chunker import Chunk


@dataclass
class Citation:
    marker: str           # e.g. "[1]"
    doc_title: str
    section_title: str
    snippet: str          # short preview of the chunk text
    relevance_score: float


def build_citations(ranked_chunks: List[Tuple[Chunk, float]]) -> List[Citation]:
    """Turns the final ranked chunks into numbered Citation objects."""
    citations = []
    for i, (chunk, score) in enumerate(ranked_chunks, start=1):
        snippet = chunk.text.strip().replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        citations.append(Citation(
            marker=f"[{i}]",
            doc_title=chunk.doc_title,
            section_title=chunk.section_title,
            snippet=snippet,
            relevance_score=round(float(score), 3),
        ))
    return citations


def format_context_for_prompt(ranked_chunks: List[Tuple[Chunk, float]]) -> str:
    """
    Formats the final chunks into a numbered text block to hand to the AI,
    e.g.:
    [1] (Document: HR Leave Policy | Section: 3. Sick Leave)
    3.1 Employees are entitled to 12 days...
    """
    blocks = []
    for i, (chunk, _) in enumerate(ranked_chunks, start=1):
        block = f"[{i}] (Document: {chunk.doc_title} | Section: {chunk.section_title})\n{chunk.text.strip()}"
        blocks.append(block)
    return "\n\n".join(blocks)