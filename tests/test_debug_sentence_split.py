import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import split_sentences, strip_boilerplate

sample_answer = """(Offline mode - no LLM configured. Showing the most relevant policy excerpts.)

[1] (Document: hr_leave_policy | Section: 5. Contact)
For questions regarding this policy, contact the HR Help Desk at hr-helpdesk@company.com."""

cleaned = strip_boilerplate(sample_answer)
print("CLEANED TEXT:")
print(repr(cleaned))
print()

sentences = split_sentences(cleaned)
print(f"Total sentences found: {len(sentences)}\n")
for i, s in enumerate(sentences, start=1):
    print(f"{i}: {repr(s)}")