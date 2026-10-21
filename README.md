# AI Policy Copilot

An educational Python RAG prototype that answers questions from policy documents using semantic search, keyword search, reranking, and citations. The repository currently includes a sample HR leave policy and is designed to support additional Insurance and Enterprise Security documents.

Ask a question such as *"How many sick days do I get?"* The system retrieves relevant policy passages and attaches source citations. When an LLM is configured, it can draft an answer such as: `Employees are entitled to 12 days of paid sick leave per year. [1]` In offline mode, it shows the retrieved excerpts instead.

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
     producing a more accurate ranking than retrieval alone
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
[5] LLM / Fallback --> if an OpenAI API key is configured, a real
     LLM drafts the answer; otherwise an offline fallback returns
     the raw top-ranked excerpts
     |
     v
Final Answer + Citations
~~~

### Why hybrid search instead of just embeddings?

Embeddings are great at matching meaning ("time off" ~ "leave") but can miss exact terms like policy clause numbers or specific codes. BM25 (keyword search) catches those. Combining both and fusing the rankings gives more robust retrieval than either alone.

### Why rerank after retrieval?

Initial retrieval (dense + sparse) is optimized for speed across large document sets, so it can return a noisy list of "roughly relevant" passages. The reranker is slower but far more accurate - it reads the question and each candidate passage together, rather than comparing pre-computed vectors. It's used only on the small shortlist, not the whole document set, to keep this fast.

## Project Structure

~~~text
ai-policy-copilot/
├── config.py                  # Central settings (chunk sizes, model names, top-k values)
├── requirements.txt           # Python dependencies
├── data/
│   ├── sample_policies/       # Source policy documents (.txt)
│   └── eval/                  # Evaluation question/answer pairs
├── scripts/
│   └── run_eval.py            # Runs the full pipeline against the eval set
├── src/
│   ├── ingestion/              # Document loading + chunking
│   ├── embeddings/             # Sentence embedding model wrapper
│   ├── retrieval/              # FAISS vector search, BM25, hybrid fusion, reranking
│   ├── citations/               # Builds numbered source citations
│   ├── generation/              # Prompt construction + LLM (or offline fallback)
│   └── evaluation/              # RAG quality metrics
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

4. (Optional) To enable real LLM-generated answers instead of the offline excerpt fallback, set an OpenAI API key as an environment variable:

~~~bash
$env:OPENAI_API_KEY="your-key-here"
~~~

Without this step, the system still works fully - it will display the top retrieved policy excerpts directly rather than an LLM-written paragraph.

## Running the Evaluation

The included evaluation harness runs the full pipeline (chunking -> embedding -> hybrid search -> reranking -> citation -> generation) against a small set of test questions and scores the results:

~~~bash
python scripts/run_eval.py
~~~

Expected output includes a per-question breakdown and aggregate scores across five metrics: context precision, context recall, faithfulness, answer relevancy, and citation validity.

## Example

**Question:** "How many sick days do I get?"

**Retrieved context (after hybrid search + reranking):**

~~~text
[1] (Document: hr_leave_policy | Section: 3. Sick Leave)
Employees are entitled to 12 days of paid sick leave per year. A medical
certificate is required for sick leave exceeding 2 consecutive working days.
~~~

**Answer:**

~~~text
Employees are entitled to 12 days of paid sick leave per year. [1]
A medical certificate is required if sick leave exceeds 2 consecutive
working days. [1]
~~~

(In offline mode - no API key configured - the system instead returns the raw numbered excerpt shown above directly, rather than an LLM-composed sentence.)

## Challenges & Debugging

A selection of real issues encountered while building this, and how they were diagnosed and fixed:

**1. Section-splitting regex over-matched sub-clauses.**
The original regex for detecting policy section headings (e.g. "3. Sick Leave") also matched numbered sub-clauses like "3.1 Employees are entitled to...", corrupting the chunk boundaries. Fixed by tightening the pattern to only match a single leading number followed by a literal period (excluding decimal-numbered lines like "3.1").

**2. Embedding similarity thresholds were miscalibrated.**
Using cosine similarity to score retrieval/faithfulness, an initial threshold of 0.45 incorrectly rated unrelated sentences as "relevant" - the embedding model used (BAAI/bge-small-en-v1.5) produces similarity scores for *unrelated* sentences in the 0.42-0.51 range, higher than commonly assumed defaults. Diagnosed with a debug script comparing known-good vs. known-bad chunk similarity, then recalibrated the threshold to 0.65, which correctly separated relevant from irrelevant context.

**3. Semantic similarity alone cannot catch numeric hallucination.**
Even after fixing the threshold, a deliberately fabricated answer ("You are entitled to 50 days of paid sick leave and a free car.") still scored a perfect 1.0 on faithfulness against context that actually said 12 days. This is because sentence embeddings capture topical/structural similarity, not precise factual correctness - a wrong number in a similarly-shaped sentence still produces a high similarity score. This is a genuine limitation of using embeddings alone to evaluate RAG systems in fact-sensitive domains like HR and Insurance.

  Fix: added a separate rule-based `numeric_consistency()` check that extracts all numbers from the answer and the retrieved context, and verifies the answer numbers are a subset of the context numbers. The final faithfulness score is now an average of the semantic score and this numeric check, which correctly dropped the hallucinated answer score from 1.0 to 0.5 while leaving accurate answers unaffected.

## Status / Roadmap

This is a working prototype covering all six core RAG concepts end-to-end (chunking, embeddings, hybrid search, reranking, citations, evaluation). Planned next steps:

- Add sample policy documents for Insurance and Enterprise Security domains (currently HR only)
- Build a Streamlit UI for interactive question-answering
- Expand the evaluation set beyond the current 4 questions
- Optional: FastAPI endpoint for programmatic access

## License

MIT License - see [LICENSE](LICENSE) for details.
