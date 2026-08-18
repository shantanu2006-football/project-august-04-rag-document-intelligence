"""Split documents into overlapping, sentence-aligned chunks for embedding."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragdoc.loaders import Document

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text produced from a Document."""

    chunk_id: str
    doc_id: str
    source_path: str
    chunk_index: int
    text: str


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    This is a lightweight heuristic, not a full sentence tokenizer: it splits
    on '.', '!', or '?' followed by whitespace and a capital letter, digit, or
    quote/paren, which works well for prose-style documents like the ones in
    this corpus without pulling in a heavyweight NLP dependency.
    """
    # Normalize whitespace (including across newlines) before splitting so a
    # paragraph break doesn't get treated as part of the sentence boundary.
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = _SENTENCE_SPLIT_RE.split(normalized)
    return [s.strip() for s in sentences if s.strip()]


def chunk_document(
    document: Document,
    max_words: int = 120,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Greedily pack a document's sentences into word-budgeted chunks.

    Sentences are appended to the current chunk until adding the next one
    would exceed max_words, at which point the chunk is closed and a new one
    is started. To preserve context across a chunk boundary, the last
    `overlap_sentences` sentences of a closed chunk are carried over as the
    start of the next chunk.

    Args:
        document: source document to split.
        max_words: soft word budget per chunk. A single sentence longer than
            this budget is still kept whole rather than being cut mid-sentence.
        overlap_sentences: number of trailing sentences repeated at the start
            of the next chunk, so a fact near a boundary isn't stranded
            without the sentence that explains it.

    Returns:
        Chunks in document order, with stable IDs of the form
        "<doc_id>::<chunk_index>".
    """
    if max_words < 1:
        raise ValueError("max_words must be >= 1")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sentences = split_sentences(document.text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    chunk_index = 0

    def flush() -> None:
        nonlocal current, current_words, chunk_index
        if not current:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::{chunk_index}",
                doc_id=document.doc_id,
                source_path=document.source_path,
                chunk_index=chunk_index,
                text=" ".join(current),
            )
        )
        chunk_index += 1

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            flush()
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += sentence_words

    flush()
    return chunks


def chunk_corpus(
    documents: list[Document],
    max_words: int = 120,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """Chunk every document in a corpus, preserving document order."""
    all_chunks: list[Chunk] = []
    for document in documents:
        all_chunks.extend(
            chunk_document(
                document, max_words=max_words, overlap_sentences=overlap_sentences
            )
        )
    return all_chunks
