from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    text: str
    score: float
