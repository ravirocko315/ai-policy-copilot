import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import numeric_consistency

good_chunks = ["3.1 Employees are entitled to 12 days of paid sick leave per year."]

good_answer = "You are entitled to 12 days of paid sick leave per year."
bad_answer = "You are entitled to 50 days of paid sick leave and a free car."

print("Good answer numeric consistency:", numeric_consistency(good_answer, good_chunks))
print("Bad answer (hallucinated number) numeric consistency:", numeric_consistency(bad_answer, good_chunks))