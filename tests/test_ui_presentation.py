import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ui.presentation import clean_answer_for_display


fallback_answer = (
    "(Offline mode - no LLM configured. Showing the most relevant policy excerpts.)\n\n"
    "[1] (Document: hr_leave_policy | Section: 3. Sick Leave)\n"
    "3.1 Employees are entitled to 12 days of paid sick leave per year.\n\n"
    "[2] (Document: insurance_claims_policy | Section: 2. Coverage)\n"
    "2.1 Annual maximum coverage is $50,000."
)

expected = (
    "3.1 Employees are entitled to 12 days of paid sick leave per year.\n\n"
    "2.1 Annual maximum coverage is $50,000."
)

actual = clean_answer_for_display(fallback_answer)
assert actual == expected, f"Unexpected cleaned answer: {actual!r}"

normal_answer = "See citation [1] for the relevant policy."
assert clean_answer_for_display(normal_answer) == normal_answer

print("PASS: offline boilerplate and source headers are removed; normal answers stay unchanged.")
