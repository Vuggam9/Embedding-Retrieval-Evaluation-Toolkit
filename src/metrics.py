from app_types import RetrievalResult



def hit_rate(expected_doc_id: str, results: list[RetrievalResult]) -> float:
    return float(any(result.doc_id == expected_doc_id for result in results))



def recall_at_k(expected_doc_id: str, results: list[RetrievalResult], k: int) -> float:
    limited_results = results[:k]
    return float(any(result.doc_id == expected_doc_id for result in limited_results))



def reciprocal_rank(expected_doc_id: str, results: list[RetrievalResult]) -> float:
    for rank, result in enumerate(results, start=1):
        if result.doc_id == expected_doc_id:
            return 1.0 / rank
    return 0.0
