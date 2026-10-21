import re
import uuid
from dataclasses import dataclass
from typing import List

from src.ingestion.loader import Document

# --- Token counting ---
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


# --- Section detection ---
SECTION_PATTERN = re.compile(
    r"(?m)^(\d+\.\s+[A-Z][^\n]{0,80})$"
)


def find_sections(text: str):
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return [("General", text)]

    sections = []
    for i, m in enumerate(matches):
        title = m.group().strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections


# --- Splitting long sections ---
def split_long_text(text: str, max_tokens: int = 350, overlap_tokens: int = 60) -> List[str]:
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    paragraphs = text.split("\n")
    units = []
    for p in paragraphs:
        if count_tokens(p) <= max_tokens:
            units.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            units.extend(sentences)

    chunks = []
    current_units = []
    current_tokens = 0

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        unit_tokens = count_tokens(unit)

        if current_tokens + unit_tokens > max_tokens and current_units:
            chunks.append(" ".join(current_units))

            overlap_units = []
            overlap_sum = 0
            for u in reversed(current_units):
                u_tokens = count_tokens(u)
                if overlap_sum + u_tokens > overlap_tokens:
                    break
                overlap_units.insert(0, u)
                overlap_sum += u_tokens

            current_units = overlap_units.copy()
            current_tokens = overlap_sum

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append(" ".join(current_units))

    return chunks


# --- Final Chunk object ---
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    category: str
    section_title: str
    text: str
    token_count: int


def chunk_document(doc: Document, max_tokens: int = 350, overlap_tokens: int = 60) -> List[Chunk]:
    chunks = []
    sections = find_sections(doc.text)

    for section_title, section_body in sections:
        pieces = split_long_text(section_body, max_tokens, overlap_tokens)
        for piece in pieces:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-{uuid.uuid4().hex[:6]}",
                doc_id=doc.doc_id,
                doc_title=doc.title,
                category=doc.category,
                section_title=section_title,
                text=piece,
                token_count=count_tokens(piece),
            ))
    return chunks


def chunk_documents(docs: List[Document], max_tokens: int = 350, overlap_tokens: int = 60) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, max_tokens, overlap_tokens))
    return all_chunks