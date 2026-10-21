from config import settings
from src.ingestion.loader import load_documents

docs = load_documents(settings.data_dir)

print(f"Loaded {len(docs)} document(s)")
for d in docs:
    print("----")
    print("Title:", d.title)
    print("Category:", d.category)
    print("First 150 characters:", d.text[:150])