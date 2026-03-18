from app_types import Chunk
from dataset import Document



def chunk_by_words(documents: list[Document], chunk_size: int = 30, overlap: int = 8) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        start = 0
        index = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(Chunk(chunk_id=f"{document.doc_id}_fixed_{index}", doc_id=document.doc_id, text=chunk_text))
            if end == len(words):
                break
            start += max(chunk_size - overlap, 1)
            index += 1
    return chunks



def chunk_by_sentences(documents: list[Document], sentences_per_chunk: int = 2) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        sentences = [sentence.strip() for sentence in document.text.split(".") if sentence.strip()]
        for index in range(0, len(sentences), max(sentences_per_chunk, 1)):
            sentence_slice = sentences[index : index + sentences_per_chunk]
            chunk_text = ". ".join(sentence_slice).strip()
            if chunk_text and not chunk_text.endswith("."):
                chunk_text += "."
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}_sentence_{index // max(sentences_per_chunk, 1)}",
                    doc_id=document.doc_id,
                    text=chunk_text,
                )
            )
    return chunks



def build_chunks(documents: list[Document], strategy: str, chunk_size: int, overlap: int) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_by_words(documents, chunk_size=chunk_size, overlap=overlap)
    if strategy == "sentence":
        sentence_window = max(chunk_size // 20, 1)
        return chunk_by_sentences(documents, sentences_per_chunk=sentence_window)
    raise ValueError(f"Unsupported chunking strategy: {strategy}")
