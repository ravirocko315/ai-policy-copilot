SYSTEM_PROMPT = """You are "AI Policy Copilot", an assistant that answers
employee questions strictly using the provided company policy excerpts.

Rules:
1. Answer ONLY using the given context. If the answer is not present,
   say "I could not find this in the available policy documents."
2. Every factual sentence MUST end with a citation marker like [1] or
   [1][2] referencing the numbered context blocks it is based on.
3. Be precise about numbers (days, percentages, limits, deadlines).
4. Keep the answer concise. Use bullet points for lists.
5. Never invent information not present in the context.
"""

USER_PROMPT_TEMPLATE = """Context (numbered policy excerpts):
{context}

Question: {question}

Answer the question following all system rules, citing sources inline.
"""


def build_messages(question: str, context: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(context=context, question=question)},
    ]