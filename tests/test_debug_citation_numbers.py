import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import faithfulness

answer = "Contact the HR Help Desk at hr-helpdesk@company.com. [1]"
context = [
    "5. Contact For any questions about this policy, contact the HR Help Desk at hr-helpdesk@company.com."
]

score = faithfulness(answer, context)
print("faithfulness score:", score)
assert score == 1.0, f"Expected 1.0, got {score}"
print("PASS: citation marker no longer counted as a fabricated number.")
