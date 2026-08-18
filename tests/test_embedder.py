from pathlib import Path

import numpy as np
import pytest

from ragdoc.embedder import TfidfSvdEmbedder

CORPUS = [
    "The cat sat on the mat and purred contentedly.",
    "A kitten played with a ball of yarn on the rug.",
    "The stock market rallied sharply after the earnings report.",
    "Quarterly revenue exceeded analyst expectations this earnings season.",
]


def test_fit_transform_shape_and_normalization():
    embedder = TfidfSvdEmbedder(n_components=2)

    vectors = embedder.fit_transform(CORPUS)

    assert vectors.shape == (len(CORPUS), embedder.dim)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(CORPUS)), atol=1e-5)


def test_n_components_is_clamped_for_small_corpora():
    # Requesting 128 components on a 4-document, few-feature corpus must not
    # raise; it should silently clamp to a value TruncatedSVD can support.
    embedder = TfidfSvdEmbedder(n_components=128)

    embedder.fit(CORPUS)

    assert 1 <= embedder.dim < len(CORPUS)


def test_similar_texts_score_higher_than_unrelated_texts():
    embedder = TfidfSvdEmbedder(n_components=2)
    vectors = embedder.fit_transform(CORPUS)

    cat_vs_kitten = float(vectors[0] @ vectors[1])
    cat_vs_stocks = float(vectors[0] @ vectors[2])

    assert cat_vs_kitten > cat_vs_stocks


def test_transform_before_fit_raises():
    embedder = TfidfSvdEmbedder(n_components=2)
    with pytest.raises(RuntimeError, match="not been fit"):
        embedder.transform(["some text"])


def test_fit_rejects_empty_corpus():
    embedder = TfidfSvdEmbedder(n_components=2)
    with pytest.raises(ValueError, match="zero texts"):
        embedder.fit([])


def test_save_and_load_round_trip_produces_identical_vectors(tmp_path: Path):
    embedder = TfidfSvdEmbedder(n_components=2)
    embedder.fit(CORPUS)
    save_path = tmp_path / "embedder.pkl"
    embedder.save(save_path)

    loaded = TfidfSvdEmbedder.load(save_path)

    original_vectors = embedder.transform(CORPUS)
    loaded_vectors = loaded.transform(CORPUS)
    np.testing.assert_allclose(original_vectors, loaded_vectors, atol=1e-6)
    assert loaded.dim == embedder.dim


def test_save_before_fit_raises(tmp_path: Path):
    embedder = TfidfSvdEmbedder(n_components=2)
    with pytest.raises(RuntimeError, match="not been fit"):
        embedder.save(tmp_path / "embedder.pkl")
