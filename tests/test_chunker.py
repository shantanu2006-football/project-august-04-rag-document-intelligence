import pytest

from ragdoc.chunker import chunk_corpus, chunk_document, split_sentences
from ragdoc.loaders import Document


def test_split_sentences_basic():
    text = "This is one sentence. This is another one! Is this a third?"
    assert split_sentences(text) == [
        "This is one sentence.",
        "This is another one!",
        "Is this a third?",
    ]


def test_split_sentences_normalizes_whitespace_and_newlines():
    text = "First sentence.\n\nSecond   sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence."]


def test_split_sentences_empty_text_returns_empty_list():
    assert split_sentences("   ") == []


def test_chunk_document_respects_word_budget():
    text = " ".join(f"Sentence number {i} has five words." for i in range(20))
    doc = Document(doc_id="doc", source_path="doc.txt", text=text)

    chunks = chunk_document(doc, max_words=20, overlap_sentences=0)

    assert len(chunks) > 1
    for chunk in chunks:
        # A single long sentence is allowed to exceed the budget, but here
        # every sentence is short, so no chunk should blow past it by much.
        assert len(chunk.text.split()) <= 20 + 6  # 6 words is one sentence


def test_chunk_document_ids_and_metadata_are_sequential():
    text = "First sentence here. Second sentence here. Third sentence here."
    doc = Document(doc_id="mydoc", source_path="path/mydoc.txt", text=text)

    chunks = chunk_document(doc, max_words=5, overlap_sentences=0)

    assert [c.chunk_id for c in chunks] == [f"mydoc::{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "mydoc" for c in chunks)
    assert all(c.source_path == "path/mydoc.txt" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_overlap_repeats_trailing_sentences():
    text = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
    doc = Document(doc_id="doc", source_path="doc.txt", text=text)

    chunks = chunk_document(doc, max_words=4, overlap_sentences=1)

    assert len(chunks) >= 2
    # The last sentence of chunk i should appear at the start of chunk i+1.
    for prev_chunk, next_chunk in zip(chunks, chunks[1:]):
        last_sentence_of_prev = split_sentences(prev_chunk.text)[-1]
        assert next_chunk.text.startswith(last_sentence_of_prev)


def test_chunk_document_empty_text_returns_no_chunks():
    doc = Document(doc_id="empty", source_path="empty.txt", text="")
    assert chunk_document(doc) == []


def test_chunk_document_rejects_invalid_params():
    doc = Document(doc_id="doc", source_path="doc.txt", text="Some text.")
    with pytest.raises(ValueError):
        chunk_document(doc, max_words=0)
    with pytest.raises(ValueError):
        chunk_document(doc, overlap_sentences=-1)


def test_chunk_corpus_preserves_document_order():
    doc_a = Document(doc_id="a", source_path="a.txt", text="A sentence about apples.")
    doc_b = Document(doc_id="b", source_path="b.txt", text="A sentence about bananas.")

    chunks = chunk_corpus([doc_a, doc_b])

    assert [c.doc_id for c in chunks] == ["a", "b"]
