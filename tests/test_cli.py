import json
from pathlib import Path

import pytest

from ragdoc.cli import main


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "cats.txt").write_text(
        "Cats are small domesticated mammals. Cats like to sleep often. "
        "Cats have retractable claws. Many cats enjoy chasing toys."
    )
    (corpus / "cars.txt").write_text(
        "Cars are motor vehicles used for transportation. Cars need fuel. "
        "Cars have four wheels typically. Many cars include safety airbags."
    )
    return corpus


def test_ingest_and_query_end_to_end(tmp_path: Path, corpus_dir: Path, capsys):
    index_dir = tmp_path / "index"

    exit_code = main(
        [
            "ingest",
            "--corpus-dir",
            str(corpus_dir),
            "--index-dir",
            str(index_dir),
            "--max-words",
            "8",
        ]
    )
    assert exit_code == 0
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "embedder.pkl").exists()

    exit_code = main(
        [
            "query",
            "What do cats like to do?",
            "--index-dir",
            str(index_dir),
            "--k",
            "1",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "chunk=cats" in out


def test_evaluate_command_end_to_end(tmp_path: Path, corpus_dir: Path, capsys):
    index_dir = tmp_path / "index"
    main(
        [
            "ingest",
            "--corpus-dir",
            str(corpus_dir),
            "--index-dir",
            str(index_dir),
            "--max-words",
            "8",
        ]
    )

    eval_file = tmp_path / "questions.json"
    eval_file.write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "What do cats like to do?",
                    "relevant_doc_ids": ["cats"],
                }
            ]
        )
    )

    exit_code = main(
        [
            "evaluate",
            "--index-dir",
            str(index_dir),
            "--eval-file",
            str(eval_file),
            "--k",
            "1",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Mean Precision@1" in out


def test_query_on_missing_index_returns_error(tmp_path: Path, capsys):
    exit_code = main(
        ["query", "anything", "--index-dir", str(tmp_path / "does_not_exist")]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()
