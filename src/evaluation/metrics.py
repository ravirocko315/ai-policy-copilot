import re
import numpy as np
from typing import List
from src.embeddings.embedder import Embedder


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]


def extract_numbers(text: str) -> set:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def numeric_consistency(answer: str, retrieved_texts: List[str]) -> float:
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return 1.0

    context_numbers = set()
    for text in retrieved_texts:
        context_numbers.update(extract_numbers(text))

    matched = answer_numbers & context_numbers
    return len(matched) / len(answer_numbers)


def context_precision(retrieved_texts: List[str], ground_truth: str, threshold: float = 0.65) -> float:
    if not retrieved_texts:
        return 0.0
    gt_vector = Embedder.get().embed([ground_truth])[0]
    chunk_vectors = Embedder.get().embed(retrieved_texts)
    relevant_count = sum(1 for v in chunk_vectors if cosine(v, gt_vector) >= threshold)
    return relevant_count / len(retrieved_texts)


def context_recall(retrieved_texts: List[str], ground_truth: str, threshold: float = 0.65) -> float:
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return 0.0

    gt_vectors = Embedder.get().embed(gt_sentences)
    context_vectors = Embedder.get().embed(retrieved_texts) if retrieved_texts else None

    covered = 0
    for gv in gt_vectors:
        if context_vectors is None:
            continue
        best_match = max(cosine(gv, cv) for cv in context_vectors)
        if best_match >= threshold:
            covered += 1
    return covered / len(gt_sentences)


def faithfulness(answer: str, retrieved_texts: List[str], threshold: float = 0.65) -> float:
    answer_sentences = split_sentences(answer)
    if not answer_sentences:
        return 0.0

    answer_vectors = Embedder.get().embed(answer_sentences)
    context_vectors = Embedder.get().embed(retrieved_texts) if retrieved_texts else None

    supported = 0
    for av in answer_vectors:
        if context_vectors is None:
            continue
        best_match = max(cosine(av, cv) for cv in context_vectors)
        if best_match >= threshold:
            supported += 1
    semantic_score = supported / len(answer_sentences)

    numeric_score = numeric_consistency(answer, retrieved_texts)

    return round((semantic_score + numeric_score) / 2, 3)


def answer_relevancy(question: str, answer: str) -> float:
    vectors = Embedder.get().embed([question, answer])
    return max(0.0, cosine(vectors[0], vectors[1]))


def citation_validity(answer: str, num_citations_available: int) -> float:
    """Checks that every [n] marker in the answer points to a citation that actually exists."""
    used_markers = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    if not used_markers:
        return 0.0
    valid = [n for n in used_markers if 1 <= n <= num_citations_available]
    return len(valid) / len(used_markers)