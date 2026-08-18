from pathlib import Path

import pytest

from ragdoc.loaders import load_corpus, load_document


def test_load_text_document(tmp_path: Path):
    path = tmp_path / "hello.txt"
    path.write_text("Hello world.\n\nThis is a test document.")

    doc = load_document(path)

    assert doc.doc_id == "hello"
    assert doc.source_path == str(path)
    assert "Hello world." in doc.text
    assert "This is a test document." in doc.text


def test_load_document_rejects_unsupported_extension(tmp_path: Path):
    path = tmp_path / "notes.docx"
    path.write_text("irrelevant")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def test_load_corpus_reads_all_supported_files_sorted(tmp_path: Path):
    (tmp_path / "b.txt").write_text("Second document.")
    (tmp_path / "a.txt").write_text("First document.")
    (tmp_path / "ignored.json").write_text('{"not": "a document"}')

    docs = load_corpus(tmp_path)

    assert [d.doc_id for d in docs] == ["a", "b"]
    assert docs[0].text == "First document."


def test_load_corpus_skips_empty_documents(tmp_path: Path):
    (tmp_path / "empty.txt").write_text("   \n\n  ")
    (tmp_path / "real.txt").write_text("Actual content.")

    docs = load_corpus(tmp_path)

    assert [d.doc_id for d in docs] == ["real"]


def test_load_corpus_raises_if_directory_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "does_not_exist")


def test_load_corpus_raises_if_no_loadable_documents(tmp_path: Path):
    (tmp_path / "notes.json").write_text("{}")

    with pytest.raises(ValueError, match="No loadable documents"):
        load_corpus(tmp_path)
