import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

@dataclass
class Settings:
    data_dir: Path = BASE_DIR / "data" / "sample_policies"
    index_dir: Path = BASE_DIR / "storage" / "index"

    max_chunk_tokens: int = 350      # how big each "note card" can be
    chunk_overlap_tokens: int = 60   # keep some overlap so we do not cut mid-sentence

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 15
    final_top_k: int = 5

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_min_score: float = 0.0

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = "openai/gpt-oss-20b"

settings = Settings()
