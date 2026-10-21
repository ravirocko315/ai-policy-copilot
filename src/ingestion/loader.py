from dataclasses import dataclass
from pathlib import Path
from typing import List
import uuid

@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    category: str

def load_documents(data_dir: Path) -> List[Document]:
    docs = []
    for path in Path(data_dir).glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        docs.append(Document(
            doc_id=str(uuid.uuid4())[:8],
            title=path.stem,
            text=text,
            category="HR",
        ))
    return docs