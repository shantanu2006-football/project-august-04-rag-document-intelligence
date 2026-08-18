from pathlib import Path

import numpy as np
import pytest

from ragdoc.chunker import Chunk
from ragdoc.vector_store import VectorStore


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        source_path=f"{chunk_id}.txt",
        chunk_index=0,
        text=f"text for {chunk_id}",
    )


def test_add_and_search_returns_closest_vector_first():
    store = VectorStore(dim=2)
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    chunks = [_make_chunk("a::0"), _make_chunk("b::0"), _make_chunk("c::0")]
    store.add(vectors, chunks)

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), k=2)

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "a::0"
    assert results[0].score > results[1].score


def test_search_k_larger_than_store_size_returns_all():
    store = VectorStore(dim=2)
    store.add(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        [_make_chunk("a::0"), _make_chunk("b::0")],
    )

    results = store.search(np.array([1.0, 0.0], dtype=np.float32), k=10)

    assert len(results) == 2


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore(dim=3)
    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5)
    assert results == []


def test_add_rejects_mismatched_lengths():
    store = VectorStore(dim=2)
    with pytest.raises(ValueError, match="vectors but"):
        store.add(np.zeros((2, 2), dtype=np.float32), [_make_chunk("a::0")])


def test_add_rejects_wrong_dim():
    store = VectorStore(dim=3)
    with pytest.raises(ValueError, match="does not match store dim"):
        store.add(np.zeros((1, 2), dtype=np.float32), [_make_chunk("a::0")])


def test_len_reflects_number_of_added_chunks():
    store = VectorStore(dim=2)
    assert len(store) == 0
    store.add(np.zeros((3, 2), dtype=np.float32), [
        _make_chunk("a::0"), _make_chunk("a::1"), _make_chunk("b::0")
    ])
    assert len(store) == 3


def test_save_and_load_round_trip(tmp_path: Path):
    store = VectorStore(dim=2)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    chunks = [_make_chunk("a::0"), _make_chunk("b::0")]
    store.add(vectors, chunks)

    save_dir = tmp_path / "index"
    store.save(save_dir)
    loaded = VectorStore.load(save_dir)

    assert len(loaded) == 2
    results = loaded.search(np.array([1.0, 0.0], dtype=np.float32), k=1)
    assert results[0].chunk.chunk_id == "a::0"
    assert results[0].chunk.text == "text for a::0"
