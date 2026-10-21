from dataclasses import dataclass
from typing import List, Tuple

from src.ingestion.chunker import Chunk
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.citations.citation_manager import build_citations, format_context_for_prompt, Citation
from src.generation.prompt_builder import build_messages
from config import settings


@dataclass
class RAGResponse:
    question: str
    answer: str
    citations: List[Citation]


class ExtractiveFallbackLLM:
    def generate(self, messages: list) -> str:
        user_msg = messages[-1]["content"]
        context_block = user_msg.split("Context (numbered policy excerpts):")[1].split("Question:")[0]
        excerpts = [b.strip() for b in context_block.strip().split("\n\n") if b.strip()]

        out = ["(Offline mode - no LLM configured. Showing the most relevant policy excerpts.)\n"]
        out.extend(excerpts)
        return "\n\n".join(out)


class OpenAILLM:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, messages: list) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=600,
        )
        return response.choices[0].message.content


def get_llm():
    if settings.openai_api_key:
        return OpenAILLM()
    return ExtractiveFallbackLLM()


class PolicyCopilot:
    def __init__(self, hybrid_searcher: HybridSearcher, reranker: Reranker = None):
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker or Reranker.get()
        self.llm = get_llm()

    def answer(self, question: str) -> RAGResponse:
        candidates = self.hybrid_searcher.search(question, top_k=settings.hybrid_top_k)
        final_chunks = self.reranker.rerank(
            question,
            candidates,
            top_k=settings.final_top_k,
            min_score=settings.rerank_min_score,
        )
        context = format_context_for_prompt(final_chunks)
        citations = build_citations(final_chunks)
        messages = build_messages(question, context)
        answer_text = self.llm.generate(messages)
        return RAGResponse(question=question, answer=answer_text, citations=citations)
