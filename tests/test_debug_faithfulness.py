import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.embedder import Embedder
from src.evaluation.metrics import cosine

good_chunk = "3.1 Employees are entitled to 12 days of paid sick leave per year."
bad_answer_sentence = "You are entitled to 50 days of paid sick leave and a free car."

vectors = Embedder.get().embed([good_chunk, bad_answer_sentence])
print("Similarity between TRUE chunk and HALLUCINATED answer:", cosine(vectors[0], vectors[1]))