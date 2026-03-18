import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class QueryExample:
    query: str
    expected_doc_id: str



def load_sample_documents() -> list[Document]:
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "sample_corpus.json"
    raw_documents = json.loads(corpus_path.read_text(encoding="utf-8"))
    return [Document(**item) for item in raw_documents]



def load_sample_queries() -> list[QueryExample]:
    return [
        QueryExample(query="Which library is used for vector similarity search?", expected_doc_id="doc_vector_search"),
        QueryExample(query="What framework helps teams build Python APIs for ML services?", expected_doc_id="doc_python_api"),
        QueryExample(query="How do embeddings represent meaning in search systems?", expected_doc_id="doc_embeddings"),
        QueryExample(query="Why do chunk boundaries matter in retrieval pipelines?", expected_doc_id="doc_chunking"),
        QueryExample(query="What helps teams monitor latency and usage in ML systems?", expected_doc_id="doc_monitoring"),
    ]
