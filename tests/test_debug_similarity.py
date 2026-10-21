import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embeddings.embedder import Embedder
from src.evaluation.metrics import cosine

ground_truth = "Employees are entitled to 12 days of paid sick leave per year."
good_chunk = "3.1 Employees are entitled to 12 days of paid sick leave per year."
bad_chunk = "For questions regarding this policy, contact the HR Help Desk."

vectors = Embedder.get().embed([ground_truth, good_chunk, bad_chunk])

print("Ground truth vs GOOD chunk similarity:", cosine(vectors[0], vectors[1]))
print("Ground truth vs BAD chunk similarity: ", cosine(vectors[0], vectors[2]))