import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import context_precision, context_recall, faithfulness, answer_relevancy

ground_truth = "Employees are entitled to 12 days of paid sick leave per year."

# Case 1: a GOOD retrieval - the chunk actually matches the ground truth
good_chunks = ["3.1 Employees are entitled to 12 days of paid sick leave per year."]

# Case 2: a BAD retrieval - totally unrelated chunk
bad_chunks = ["For questions regarding this policy, contact the HR Help Desk."]

print("=== Context Precision ===")
print("Good retrieval:", context_precision(good_chunks, ground_truth))
print("Bad retrieval: ", context_precision(bad_chunks, ground_truth))

print("\n=== Faithfulness ===")
good_answer = "You are entitled to 12 days of paid sick leave per year."
bad_answer = "You are entitled to 50 days of paid sick leave and a free car."

print("Good answer vs good context:", faithfulness(good_answer, good_chunks))
print("Bad answer (hallucinated) vs good context:", faithfulness(bad_answer, good_chunks))

print("\n=== Answer Relevancy ===")
question = "How many sick days do I get?"
print("Relevant answer:", answer_relevancy(question, good_answer))
print("Irrelevant answer:", answer_relevancy(question, "The weather is nice today."))