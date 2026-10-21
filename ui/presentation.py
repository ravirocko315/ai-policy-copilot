import re


def clean_answer_for_display(answer: str) -> str:
    """Clean the offline excerpt display without changing normal AI answers."""
    if not answer.startswith("(Offline mode"):
        return answer

    cleaned_lines = []
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.startswith("(Offline mode"):
            continue
        if re.fullmatch(r"\[\d+\]\s*\(Document:.*\)", stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
