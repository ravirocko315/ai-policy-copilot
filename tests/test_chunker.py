from config import settings
from src.ingestion.loader import load_documents
from src.ingestion.chunker import find_sections

docs = load_documents(settings.data_dir)
sections = find_sections(docs[0].text)

print(f"Found {len(sections)} sections\n")
for title, body in sections:
    print("SECTION TITLE:", title)
    print("BODY PREVIEW:", body[:100].replace("\n", " "))
    print("---")