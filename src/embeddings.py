from sentence_transformers import SentenceTransformer



def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)



def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.tolist()
