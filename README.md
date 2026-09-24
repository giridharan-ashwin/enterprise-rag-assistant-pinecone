# Enterprise RAG Assistant

A production-oriented Retrieval-Augmented Generation (RAG) system using **OpenAI, Pinecone, FastAPI, dense retrieval, BM25 sparse retrieval, reciprocal rank fusion, evidence selection, and grounded answer generation**.

The repository also contains the reproducible benchmark and evaluation artifacts used in the research study:

> **Evaluating Dense, Hybrid, and Threshold-Based Retrieval Strategies for Enterprise RAG Systems**

## Overview

The project evaluates how retrieval configuration affects:

- supported-query retrieval accuracy
- unsupported-query rejection
- multi-section evidence coverage
- downstream answer correctness
- answer faithfulness
- citation accuracy
- abstention behavior

## Architecture

```text
User Query
  |
  v
FastAPI
  |
  v
RAG Service
  |
  +----------------------+
  |                      |
  v                      v
Dense Retrieval        BM25 Retrieval
(Pinecone)             (Local)
  |                      |
  +----------+-----------+
         |
         v
   Reciprocal Rank Fusion
         |
         v
    Evidence Selection
         |
         v
    Optional Reranking
         |
         v
    OpenAI Generation
         |
         v
   Grounded Answer
   + Citations
   + Abstention
   + Metrics
```

## Vector Configuration

- Pinecone index: `enterprise-rag`
- Metric: cosine
- Dimension: 512
- Embedding model: `text-embedding-3-small`
- Embedding dimensions: 512
- Chat model: `gpt-4o-mini`
- Default top-k: 5
- Chunk size: 900
- Chunk overlap: 120
- Research namespace: `stress-test`
- Dense similarity threshold: 0.50
- RRF constant: 60

## Requirements

- Python 3.11+
- OpenAI API key
- Pinecone API key
- Pinecone index configured with:
  - dimension: 512
  - metric: cosine

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add the required API keys to `.env`. Do not commit `.env`.

### Running the Application

Ingest the sample handbook:

```bash
python -m app.ingest data/sample/company_handbook.md
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open Swagger:

```
http://localhost:8000/docs
```

### Example Query

```json
{
  "question": "What is the remote work policy?",
  "top_k": 5
}
```

The API returns:
- generated answer
- citations
- retrieved contexts
- abstention status
- latency metrics
- token usage
- estimated generation cost

## Research Dataset

The research benchmark uses a synthetic enterprise-style corpus containing:

- 20 enterprise documents
- approximately 100 indexed chunks
- 150 retrieval evaluation cases
- 135 supported questions
- 15 unsupported questions
- 15 multi-section questions

The downstream answer-quality benchmark contains:

- 30 supported single-section questions
- 15 multi-section questions
- 15 unsupported questions
- 60 total cases

The dataset is synthetic and is intended for controlled experimentation rather than representation of a specific real organization.

## Research Experiments

The repository evaluates several retrieval configurations:

- Dense retrieval
- Dense retrieval with similarity-based evidence selection
- Hybrid dense + BM25 retrieval
- Hybrid retrieval with threshold-based evidence selection
- Exploratory hybrid retrieval with thresholding and reranking

The primary research comparison focuses on the interaction between hybrid retrieval and evidence selection.

## Evaluation Commands

| Command | Purpose |
|---------|---------|
| `python tests/research_benchmark.py` | Retrieval Benchmark |
| `python tests/hybrid_stress_test.py` | Hybrid Stress Test |
| `python tests/threshold_eval.py` | Threshold Evaluation |
| `python tests/threshold_sweep.py` | Threshold Sweep |
| `python tests/answer_eval.py` | Answer Evaluation |
| `python tests/answer_eval_summary.py` | Answer Quality Summary |
| `python tests/answer_judge.py` | LLM-Based Answer Judge |
| `python tests/inspect_answer_failures.py` | Inspect Answer Failures |

## Research Artifacts

Research documentation is located under `research/`:

- `research_questions.md`
- `methodology.md`
- `dataset.md`
- `experiments.md`
- `limitations.md`
- `paper_outline.md`
- `paper.md`

Evaluation artifacts are stored under `results/`:

- `hybrid_stress_results.json`
- `retrieval_cache.json`
- `answer_eval_results.json`
- `answer_eval_summary.json`
- `answer_judge_results.json`

## Reproducibility

The reported research results depend on the following configuration:

- OpenAI embedding model: `text-embedding-3-small`
- Embedding dimension: 512
- OpenAI generation model: `gpt-4o-mini`
- Pinecone similarity metric: cosine
- Hybrid retrieval: dense + BM25
- Reciprocal Rank Fusion constant: 60
- Retrieval top-k: 5
- Dense similarity threshold: 0.50
- Synthetic enterprise corpus
- Fixed benchmark case files under `tests/`

API keys and service credentials are intentionally excluded from the repository.

Because the benchmark uses external model and retrieval services, exact generation behavior may vary if model versions or external service behavior changes.

## Repository Structure

```
app/
  main.py
  config.py
  clients.py
  models.py
  ingest.py
  services/
  chunker.py
  embeddings.py
  ingestion.py
  retrieval.py
  generation.py
  rag.py
  hybrid.py

data/
  sample/
  stress_test/

tests/
  retrieval and answer evaluation scripts

research/
  research methodology and manuscript

results/
  evaluation artifacts

scripts/
  supporting ingestion scripts
```

## Limitations

The benchmark has several limitations:

- synthetic enterprise corpus
- limited benchmark size
- single primary embedding model
- single primary generation model
- external-service dependency
- limited unsupported-query sample
- limited multi-section sample
- LLM-based answer judging
- exploratory rather than controlled reranking evaluation

These limitations are discussed in detail in `research/limitations.md`.

## Research Paper

The complete manuscript is available at: `research/paper.md`

## License

This repository is intended for research, experimentation, and demonstration purposes.
