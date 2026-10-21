import json
from pathlib import Path
from typing import List, Dict

from src.generation.rag_pipeline import PolicyCopilot
from src.evaluation.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    citation_validity,
)


def load_eval_set(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(copilot: PolicyCopilot, eval_set: List[Dict]) -> List[Dict]:
    """Runs every question through the pipeline and scores it against all 5 metrics."""
    rows = []

    for item in eval_set:
        question = item["question"]
        ground_truth = item["ground_truth"]

        response = copilot.answer(question)
        retrieved_texts = [c.snippet for c in response.citations]

        row = {
            "question": question,
            "answer": response.answer[:150].replace("\n", " ") + "...",
            "context_precision": round(context_precision(retrieved_texts, ground_truth), 3),
            "context_recall": round(context_recall(retrieved_texts, ground_truth), 3),
            "faithfulness": round(faithfulness(response.answer, retrieved_texts), 3),
            "answer_relevancy": round(answer_relevancy(question, response.answer), 3),
            "citation_validity": round(citation_validity(response.answer, len(response.citations)), 3),
        }
        rows.append(row)

    return rows


def print_report(rows: List[Dict]):
    print("\n" + "=" * 100)
    print("RAG EVALUATION REPORT")
    print("=" * 100)

    metric_keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "citation_validity"]

    for row in rows:
        print(f"\nQ: {row['question']}")
        print(f"A: {row['answer']}")
        for k in metric_keys:
            print(f"   {k:20s}: {row[k]}")

    print("\n" + "-" * 100)
    print("AGGREGATE SCORES (average across all questions)")
    print("-" * 100)
    for k in metric_keys:
        avg = sum(r[k] for r in rows) / len(rows)
        print(f"   {k:20s}: {round(avg, 3)}")
    print("=" * 100)