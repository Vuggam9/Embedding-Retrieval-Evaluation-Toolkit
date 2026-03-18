import argparse
from pathlib import Path

import pandas as pd

from chunking import build_chunks
from dataset import load_sample_documents, load_sample_queries
from embeddings import embed_texts, load_embedding_model
from metrics import hit_rate, recall_at_k, reciprocal_rank
from reporting import save_reports
from retrieval import build_faiss_index, search_index


DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/paraphrase-MiniLM-L3-v2",
]
DEFAULT_STRATEGIES = ["fixed", "sentence"]



def evaluate_configuration(model_name: str, strategy: str, chunk_size: int, overlap: int, top_k: int) -> dict[str, object]:
    documents = load_sample_documents()
    queries = load_sample_queries()
    chunks = build_chunks(documents, strategy=strategy, chunk_size=chunk_size, overlap=overlap)

    model = load_embedding_model(model_name)
    chunk_embeddings = embed_texts(model, [chunk.text for chunk in chunks])
    index = build_faiss_index(chunk_embeddings)

    hit_scores: list[float] = []
    recall_scores: list[float] = []
    reciprocal_ranks: list[float] = []

    for query in queries:
        query_embedding = embed_texts(model, [query.query])[0]
        results = search_index(index, query_embedding, chunks, top_k=top_k)
        hit_scores.append(hit_rate(query.expected_doc_id, results))
        recall_scores.append(recall_at_k(query.expected_doc_id, results, top_k))
        reciprocal_ranks.append(reciprocal_rank(query.expected_doc_id, results))

    return {
        "model_name": model_name,
        "chunking_strategy": strategy,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "top_k": top_k,
        "num_chunks": len(chunks),
        "num_queries": len(queries),
        "hit_rate": round(sum(hit_scores) / len(hit_scores), 3),
        "recall_at_k": round(sum(recall_scores) / len(recall_scores), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }



def run_benchmarks(models: list[str], strategies: list[str], top_k: int) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    chunk_configs = [
        {"chunk_size": 20, "overlap": 5},
        {"chunk_size": 30, "overlap": 8},
        {"chunk_size": 45, "overlap": 10},
    ]

    for model_name in models:
        for strategy in strategies:
            for config in chunk_configs:
                results.append(
                    evaluate_configuration(
                        model_name=model_name,
                        strategy=strategy,
                        chunk_size=config["chunk_size"],
                        overlap=config["overlap"],
                        top_k=top_k,
                    )
                )

    return pd.DataFrame(results)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding and retrieval benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    results_df = run_benchmarks(models=args.models, strategies=args.strategies, top_k=args.top_k)
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    csv_path, md_path = save_reports(results_df, reports_dir)
    print(results_df.to_string(index=False))
    print(f"\nSaved CSV report to: {csv_path}")
    print(f"Saved Markdown summary to: {md_path}")


if __name__ == "__main__":
    main()
