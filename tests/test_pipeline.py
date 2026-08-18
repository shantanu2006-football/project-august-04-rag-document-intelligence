from pathlib import Path

import pytest

from ragdoc.pipeline import RagPipeline

CATS_TEXT = (
    "Cats are small domesticated carnivorous mammals. "
    "Cats spend most of their day sleeping. "
    "A group of cats is called a clowder."
)
COOKING_TEXT = (
    "Sauteing is a cooking technique that uses a small amount of fat in a pan. "
    "High heat and constant motion keep food from sticking. "
    "It is one of the fastest ways to cook vegetables."
)


def _build_corpus(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "cats.txt").write_text(CATS_TEXT)
    (corpus_dir / "cooking.txt").write_text(COOKING_TEXT)
    return corpus_dir


def test_build_indexes_all_chunks(tmp_path: Path):
    corpus_dir = _build_corpus(tmp_path)

    pipeline = RagPipeline.build(corpus_dir, max_words=20, overlap_sentences=0)

    assert len(pipeline.vector_store) > 0


def test_retrieve_returns_relevant_document_first(tmp_path: Path):
    corpus_dir = _build_corpus(tmp_path)
    pipeline = RagPipeline.build(corpus_dir, max_words=20, overlap_sentences=0)

    results = pipeline.retrieve("What do cats spend their day doing?", k=2)

    assert results[0].chunk.doc_id == "cats"


def test_answer_extracts_relevant_sentence(tmp_path: Path):
    corpus_dir = _build_corpus(tmp_path)
    pipeline = RagPipeline.build(corpus_dir, max_words=20, overlap_sentences=0)

    answer = pipeline.answer("What is sauteing?", k=2)

    assert "sauteing" in answer.best_sentence.lower()
    assert len(answer.sources) > 0


def test_answer_with_no_results_returns_empty_answer(tmp_path: Path):
    corpus_dir = _build_corpus(tmp_path)
    pipeline = RagPipeline.build(corpus_dir, max_words=20, overlap_sentences=0)

    from ragdoc.answerer import extract_answer

    answer = extract_answer("irrelevant", [], pipeline.embedder)

    assert answer.best_sentence == ""
    assert answer.sources == []


def test_save_and_load_round_trip_preserves_retrieval(tmp_path: Path):
    corpus_dir = _build_corpus(tmp_path)
    pipeline = RagPipeline.build(corpus_dir, max_words=20, overlap_sentences=0)
    index_dir = tmp_path / "index"
    pipeline.save(index_dir)

    loaded = RagPipeline.load(index_dir)
    results = loaded.retrieve("cats sleeping", k=1)

    assert len(loaded.vector_store) == len(pipeline.vector_store)
    assert results[0].chunk.doc_id == "cats"


def test_build_raises_on_empty_corpus_dir(tmp_path: Path):
    empty_dir = tmp_path / "empty_corpus"
    empty_dir.mkdir()

    with pytest.raises(ValueError):
        RagPipeline.build(empty_dir)
