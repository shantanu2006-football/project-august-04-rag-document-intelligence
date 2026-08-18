import json
from pathlib import Path

import pytest

from ragdoc.evaluation import (
    QuestionExample,
    _score_question,
    evaluate_pipeline,
    load_question_set,
)
from ragdoc.pipeline import RagPipeline


def test_score_question_perfect_match_at_rank_one():
    example = QuestionExample(id="q1", question="?", relevant_doc_ids=["a"])
    result = _score_question(example, retrieved_doc_ids=["a", "b", "c"], k=3)

    assert result.precision_at_k == pytest.approx(1 / 3)
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0


def test_score_question_relevant_doc_at_rank_two():
    example = QuestionExample(id="q1", question="?", relevant_doc_ids=["b"])
    result = _score_question(example, retrieved_doc_ids=["a", "b", "c"], k=3)

    assert result.reciprocal_rank == pytest.approx(0.5)


def test_score_question_no_relevant_doc_retrieved():
    example = QuestionExample(id="q1", question="?", relevant_doc_ids=["z"])
    result = _score_question(example, retrieved_doc_ids=["a", "b", "c"], k=3)

    assert result.precision_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0


def test_score_question_multiple_relevant_docs_partial_recall():
    example = QuestionExample(id="q1", question="?", relevant_doc_ids=["a", "z"])
    result = _score_question(example, retrieved_doc_ids=["a", "b", "c"], k=3)

    assert result.recall_at_k == pytest.approx(0.5)


def test_load_question_set(tmp_path: Path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [{"id": "q1", "question": "What?", "relevant_doc_ids": ["doc1"]}]
        )
    )

    questions = load_question_set(path)

    assert len(questions) == 1
    assert questions[0].id == "q1"
    assert questions[0].relevant_doc_ids == ["doc1"]


def test_load_question_set_rejects_empty_file(tmp_path: Path):
    path = tmp_path / "questions.json"
    path.write_text("[]")

    with pytest.raises(ValueError, match="empty"):
        load_question_set(path)


def test_evaluate_pipeline_end_to_end(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "cats.txt").write_text(
        "Cats are small domesticated mammals. Cats like to sleep often. "
        "Cats have retractable claws. Many cats enjoy chasing toys."
    )
    (corpus_dir / "cars.txt").write_text(
        "Cars are motor vehicles used for transportation. Cars need fuel or batteries. "
        "Cars have four wheels typically. Many cars include safety airbags."
    )
    pipeline = RagPipeline.build(corpus_dir, max_words=8, overlap_sentences=0)

    questions = [
        QuestionExample(
            id="q1", question="What do cats like to do?", relevant_doc_ids=["cats"]
        ),
        QuestionExample(
            id="q2", question="What do cars need to run?", relevant_doc_ids=["cars"]
        ),
    ]

    report = evaluate_pipeline(pipeline, questions, k=1)

    assert report.n_questions == 2
    assert report.mean_reciprocal_rank == 1.0
    assert report.hit_rate_at_k == 1.0
