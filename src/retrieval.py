import faiss
import numpy as np

from app_types import Chunk, RetrievalResult



def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
    vectors = np.array(embeddings, dtype="float32")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index



def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: list[float],
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[RetrievalResult]:
    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, top_k)
    results: list[RetrievalResult] = []
    for score, index_value in zip(scores[0], indices[0]):
        if index_value == -1:
            continue
        chunk = chunks[index_value]
        results.append(
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(score),
            )
        )
    return results
