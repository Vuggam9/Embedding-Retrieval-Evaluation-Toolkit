from app_types import RetrievalResult
from metrics import hit_rate, recall_at_k, reciprocal_rank



def make_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(chunk_id="a", doc_id="wrong_doc", text="x", score=0.4),
        RetrievalResult(chunk_id="b", doc_id="target_doc", text="y", score=0.3),
    ]



def test_hit_rate_returns_one_when_doc_found() -> None:
    assert hit_rate("target_doc", make_results()) == 1.0



def test_recall_at_k_checks_top_window() -> None:
    assert recall_at_k("target_doc", make_results(), 1) == 0.0
    assert recall_at_k("target_doc", make_results(), 2) == 1.0



def test_reciprocal_rank_rewards_higher_rank() -> None:
    assert reciprocal_rank("target_doc", make_results()) == 0.5
