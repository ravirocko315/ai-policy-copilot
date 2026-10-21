# AI Policy Copilot

An educational Python RAG prototype that answers questions from policy documents using semantic search, keyword search, reranking, and citations. The repository includes sample policy documents across three domains: HR (leave policy), Insurance (claims policy), and Enterprise Security (information security policy). An interactive Streamlit web UI is included for trying it out directly in the browser (see Running the UI below).

Ask a question such as *"How many sick days do I get?"* The system retrieves relevant policy passages and attaches source citations. When an LLM is configured (OpenAI or Groq), it drafts an answer such as: `Employees are entitled to 12 days of paid sick leave per year. [1]` Without an LLM key configured, it shows the retrieved excerpts directly instead.

## Why this project exists

A basic "chatbot over documents" often embeds text, searches it, and sends the results to an LLM. This project explores additional techniques for more reliable retrieval in policy settings, where incorrect details or numbers can matter:

- **Chunking** to split documents into useful, section-aware passages
- **Embeddings** to find passages with meaning related to a question
- **Hybrid search** to combine semantic matching with keyword matching
- **Reranking** to reorder retrieved passages by relevance
- **Citations** to connect responses to source passages
- **RAG evaluation** to measure retrieval and answer quality, including a numeric-consistency check that complements semantic similarity

## How It Works

A question goes through the following pipeline:

~~~text
User Question
     |
     v
[1] Hybrid Search --> combines two retrieval methods:
     - Dense (FAISS + sentence embeddings): finds passages with
       similar MEANING, even if wording differs
     - Sparse (BM25): finds passages with matching KEYWORDS, good
       for exact terms like section numbers or codes
     Results are merged using Reciprocal Rank Fusion
     |
     v
[2] Reranking --> a cross-encoder model re-scores the merged
     candidates by reading the question and each passage together,
     producing a more accurate ranking than retrieval alone, and
     filters out candidates below a minimum relevance score
     |
     v
[3] Citation Builder --> attaches a numbered source reference
     ([1], [2]...) to each retrieved passage, tracking document
     title and section
     |
     v
[4] Prompt Builder --> assembles the question + numbered context
     into a strict prompt instructing the LLM to only answer from
     the given context and cite every claim
     |
     v
[5] LLM / Fallback --> if a Groq or OpenAI API key is configured,
     a real LLM drafts the answer; otherwise an offline fallback
     returns the raw top-ranked excerpts
     |
     v
Final Answer + Citations
~~~

### Why hybrid search instead of just embeddings?

Embeddings are great at matching meaning ("time off" ~ "leave") but can miss exact terms like policy clause numbers or specific codes. BM25 (keyword search) catches those. Combining both and fusing the rankings gives more robust retrieval than either alone.

### Why rerank after retrieval?

Initial retrieval (dense + sparse) is optimized for speed across large document sets, so it can return a noisy list of "roughly relevant" passages. The reranker is slower but far more accurate - it reads the question and each candidate passage together, rather than comparing pre-computed vectors. This is used only on the small shortlist, not the whole document set, to keep things fast. It also filters out passages that score below a relevance threshold, rather than always returning a fixed number of results (see Challenge #4 below).

## Project Structure

~~~text
ai-policy-copilot/
├── config.py                  # Central settings (chunk sizes, model names, top-k values)
├── requirements.txt           # Python dependencies
├── .env                        # API keys (not committed, see Setup below)
├── data/
│   ├── sample_policies/       # Source policy documents (.txt) - HR, Insurance, Security
│   └── eval/                  # Evaluation question/answer pairs
├── scripts/
│   └── run_eval.py            # Runs the full pipeline against the eval set
├── src/
│   ├── ingestion/              # Document loading + chunking
│   ├── embeddings/             # Sentence embedding model wrapper
│   ├── retrieval/              # FAISS vector search, BM25, hybrid fusion, reranking
│   ├── citations/               # Builds numbered source citations
│   ├── generation/              # Prompt construction + LLM (Groq, OpenAI, or offline fallback)
│   └── evaluation/              # RAG quality metrics
├── ui/
│   ├── app.py                  # Streamlit web interface
│   └── presentation.py         # Helpers for cleaning answer text for display
└── tests/                      # Isolated test scripts for each component
~~~

## Setup

**Prerequisites:** Python 3.11 recommended (3.10+ should work), pip, and roughly 2GB of free disk space (for ML model downloads on first run).

1. Clone the repository:

~~~bash
git clone https://github.com/<your-username>/ai-policy-copilot.git
cd ai-policy-copilot
~~~

2. Create and activate a virtual environment:

~~~bash
python -m venv venv
venv\Scripts\activate
~~~

3. Install dependencies:

~~~bash
pip install -r requirements.txt
~~~

4. (Optional) To enable real LLM-generated answers instead of the offline excerpt fallback, create a file named `.env` in the project root with one of the following:

~~~text
GROQ_API_KEY=your-groq-key-here
~~~

or

~~~text
OPENAI_API_KEY=your-openai-key-here
~~~

Groq is free to use and is checked first if both keys are present. Get a Groq key at https://console.groq.com. The `.env` file is excluded from version control by `.gitignore` and is loaded automatically at startup.

Without this step, the system still works fully - it will display the top retrieved policy excerpts directly rather than an LLM-written paragraph.

## Running the UI

A Streamlit web interface is included for interactively asking questions without using the command line.

~~~bash
streamlit run ui\app.py
~~~

This opens a browser window at http://localhost:8501 with:

- A sidebar with a project description and example questions you can click to try instantly
- A text box to type your own question
- An Answer section showing the retrieved or generated answer
- A Sources section listing each cited policy passage as an expandable box, showing the document, section, and reranker relevance score

The first load takes 10-30 seconds while the embedding and reranker models load and the document index is built. This is cached afterward, so follow-up questions are fast.

## Running the Evaluation

The included evaluation harness runs the full pipeline (chunking -> embedding -> hybrid search -> reranking -> citation -> generation) against a set of test questions spanning all three policy domains and scores the results:

~~~bash
python scripts/run_eval.py
~~~

Expected output includes a per-question breakdown and aggregate scores across five metrics: context precision, context recall, faithfulness, answer relevancy, and citation validity.

**Current aggregate scores (8 questions across 3 documents, using Groq for generation):**

~~~text
context_precision   : 0.958
context_recall       : 1.0
faithfulness          : 1.0
answer_relevancy      : 0.867
citation_validity     : 1.0
~~~

## Example

**Question:** "How many sick days do I get?"

**Retrieved context (after hybrid search + reranking):**

~~~text
[1] (Document: hr_leave_policy | Section: 3. Sick Leave)
Employees are entitled to 12 days of paid sick leave per year. A medical
certificate is required for sick leave exceeding 2 consecutive working days.
~~~

**Answer (generated by Groq, llama-family model):**

~~~text
You get 12 days of paid sick leave per year. [1]
~~~

(Without an LLM key configured, the system instead returns the raw numbered excerpt shown above directly, rather than an LLM-composed sentence.)

## Challenges & Debugging

A selection of real issues encountered while building this, and how they were diagnosed and fixed:

**1. Section-splitting regex over-matched sub-clauses.**
The original regex for detecting policy section headings (e.g. "3. Sick Leave") also matched numbered sub-clauses like "3.1 Employees are entitled to...", corrupting the chunk boundaries. Fixed by tightening the pattern to only match a single leading number followed by a literal period (excluding decimal-numbered lines like "3.1").

**2. Embedding similarity thresholds were miscalibrated.**
Using cosine similarity to score retrieval/faithfulness, an initial threshold of 0.45 incorrectly rated unrelated sentences as "relevant" - the embedding model used (BAAI/bge-small-en-v1.5) produces similarity scores for *unrelated* sentences in the 0.42-0.51 range, higher than commonly assumed defaults. Diagnosed with a debug script comparing known-good vs. known-bad chunk similarity, then recalibrated the threshold to 0.65, which correctly separated relevant from irrelevant context.

**3. Semantic similarity alone cannot catch numeric hallucination.**
Even after fixing the threshold, a deliberately fabricated answer ("You are entitled to 50 days of paid sick leave and a free car.") still scored a perfect 1.0 on faithfulness against context that actually said 12 days. This is because sentence embeddings capture topical/structural similarity, not precise factual correctness - a wrong number in a similarly-shaped sentence still produces a high similarity score. This is a genuine limitation of using embeddings alone to evaluate RAG systems in fact-sensitive domains like HR and Insurance.

  Fix: added a separate rule-based `numeric_consistency()` check that extracts all numbers from the answer and the retrieved context, and verifies the answer numbers are a subset of the context numbers. The final faithfulness score is now an average of the semantic score and this numeric check, which correctly dropped the hallucinated answer score from 1.0 to 0.5 while leaving accurate answers unaffected.

**4. Fixed top-k retrieval padded results with irrelevant chunks.**
After expanding the corpus from one document to three (HR, Insurance, Enterprise Security), context_precision unexpectedly dropped from 0.55 to 0.4 despite retrieval accuracy actually being correct. A diagnostic script printing reranker scores per chunk revealed the real cause: the reranker correctly identified exactly one relevant chunk per question (scoring strongly positive, e.g. +4 to +7), while the remaining 4 chunks returned to satisfy a fixed top_k=5 were clearly irrelevant (scoring strongly negative, e.g. -6 to -10). The system was being forced to pad results with filler just to hit a fixed count. Fixed by adding a `min_score` cutoff to the reranker, so it returns only chunks above a relevance threshold (falling back to the single best match if none qualify) instead of always returning exactly k results. This raised context_precision to 0.958.

**5. Boilerplate formatting corrupted the faithfulness metric.**
After the above fix reduced each answer to a single cited excerpt, faithfulness unexpectedly dropped (0.721 to 0.572), with the worst scores on the simplest, most accurate answers. A debug script printing the sentence-splitter output revealed the cause: citation headers like `[1] (Document: hr_leave_policy | Section: 5. Contact)` contain periods that the sentence-splitter misinterpreted as sentence boundaries, chopping headers into meaningless fragments that could never semantically match real context. Fixed by adding a `strip_boilerplate()` step that removes offline-mode notices and citation headers before scoring faithfulness, while leaving the user-facing answer untouched. This raised faithfulness to 0.885 - higher than the original single-document baseline (0.842), despite the corpus being three times larger and spanning three distinct domains.

**6. A real LLM citation marker was mistaken for a fabricated number.**
After connecting a real LLM (Groq) instead of the offline fallback, faithfulness unexpectedly dropped again, worst on the simplest answers - e.g. "Contact the HR Help Desk at hr-helpdesk@company.com. [1]" scored only 0.5, despite being word-for-word correct. The `numeric_consistency()` check extracts every number from an answer and verifies it appears in the retrieved context. Since this answer had no real facts to check, the only "number" it found was the 1 inside the inline citation marker `[1]` - which does not appear anywhere in the context text itself (a contact section has no numbers), so the check flagged it as an invented fact. Fixed by extending `strip_boilerplate()` to remove inline `[N]` citation markers before numeric scoring (citation display to the user is unaffected, since this stripping only happens inside the evaluation metric, not the UI). This raised faithfulness to a perfect 1.0 across all 8 questions when evaluated end-to-end with a real LLM.

## Status / Roadmap

This is a working prototype covering all six core RAG concepts end-to-end (chunking, embeddings, hybrid search, reranking, citations, evaluation), with sample documents across HR, Insurance, and Enterprise Security domains, an 8-question evaluation set spanning all three, an interactive Streamlit UI, and real LLM-generated answers via Groq (OpenAI also supported). Planned next steps:

- Expand the evaluation set further
- Optional: FastAPI endpoint for programmatic access

## License

MIT License - see [LICENSE](LICENSE) for details.
