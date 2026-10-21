from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents

docs = load_documents(settings.data_dir)
chunks = chunk_documents(docs, max_tokens=settings.max_chunk_tokens, overlap_tokens=settings.chunk_overlap_tokens)

print(f"Loaded {len(docs)} document(s) -> produced {len(chunks)} chunk(s)\n")

for c in chunks:
    print(f"[{c.section_title}] ({c.token_count} tokens)")
    print(c.text[:120].replace("\n", " "), "...")
    print("---")