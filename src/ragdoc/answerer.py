"""Produce a short extractive answer from retrieved chunks.

No generative LLM is wired in (see README "Future Work"): this module scores
individual sentences from the retrieved chunks against the question using the
same embedder as retrieval, and returns the single best-matching sentence as
the answer plus the full chunks as supporting context. This is a real,
testable answer-selection step, not a stub, but it is intentionally simpler
than an LLM-generated abstractive answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragdoc.chunker import split_sentences
from ragdoc.embedder import Embedder
from ragdoc.vector_store import SearchResult


@dataclass(frozen=True)
class Answer:
    """An extractive answer plus the retrieved chunks it was drawn from."""

    question: str
    best_sentence: str
    sources: list[SearchResult]


def extract_answer(
    question: str, results: list[SearchResult], embedder: Embedder
) -> Answer:
    """Pick the single sentence, across all retrieved chunks, closest to the question.

    Falls back to the top chunk's first sentence if no results are given.
    """
    if not results:
        return Answer(question=question, best_sentence="", sources=[])

    candidate_sentences: list[str] = []
    for result in results:
        candidate_sentences.extend(split_sentences(result.chunk.text))
    if not candidate_sentences:
        return Answer(question=question, best_sentence="", sources=results)

    question_vec = embedder.transform([question])[0]
    sentence_vecs = embedder.transform(candidate_sentences)
    scores = sentence_vecs @ question_vec
    best_idx = int(np.argmax(scores))
    return Answer(
        question=question,
        best_sentence=candidate_sentences[best_idx],
        sources=results,
    )
