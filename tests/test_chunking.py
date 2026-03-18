from chunking import build_chunks
from dataset import load_sample_documents



def test_fixed_chunking_returns_chunks() -> None:
    chunks = build_chunks(load_sample_documents(), strategy="fixed", chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(chunk.doc_id for chunk in chunks)



def test_sentence_chunking_returns_chunks() -> None:
    chunks = build_chunks(load_sample_documents(), strategy="sentence", chunk_size=20, overlap=5)
    assert len(chunks) > 0
    assert all(chunk.text.endswith(".") for chunk in chunks)
