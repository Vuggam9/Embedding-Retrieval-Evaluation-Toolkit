# Embedding & Retrieval Evaluation Toolkit

This project came from a practical question: if retrieval quality is weak in a RAG system, is the issue the embedding model, the chunking strategy, or the ranking setup? The toolkit is meant to make those comparisons easier to run and easier to explain.

## What This Project Explores

It demonstrates:

- reproducible benchmark design
- configurable chunking strategies
- FAISS-based vector search
- comparison of multiple embedding models
- evaluation metrics such as Hit Rate, Recall@K, and MRR
- report generation for experiment review
- notebook-based exploration plus automated tests

## Setup

```powershell
cd "C:\Users\Maneesha Vuggam\Documents\New project\embedding-retrieval-eval-toolkit"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run A Benchmark

```powershell
python src/benchmark.py --top-k 3
```

## Outputs

The benchmark writes:

- `reports/benchmark_results.csv`
- `reports/benchmark_summary.md`

## Tradeoffs And Limitations

- The included dataset is intentionally small so the project stays easy to run locally.
- FAISS is a solid local baseline, but this project does not benchmark a hosted vector database.
- The metrics focus on retrieval quality rather than downstream answer quality from an LLM.

## Why I Built It

I wanted a project that sat one layer below a full RAG application. Instead of only showing a final chatbot-style experience, this project makes the retrieval system measurable, which is often the part that needs tuning first.

## Resume Bullet

Built an embedding and retrieval evaluation toolkit using FAISS and Sentence Transformers to benchmark semantic search quality across embedding models and chunking strategies, generating reproducible reports with Recall@K and MRR for retrieval tuning workflows.
