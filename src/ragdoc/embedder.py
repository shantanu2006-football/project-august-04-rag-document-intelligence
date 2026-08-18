"""Turn chunk text into dense vectors suitable for FAISS cosine search.

The default embedder is TF-IDF followed by truncated SVD (i.e. classic LSA):
it needs no downloaded model weights, is fully deterministic, and fits/embeds
a corpus of this size in milliseconds, which keeps ingestion, tests, and CI
fast and fully offline. See README "Design Decisions" for the trade-off
against a neural sentence-embedding model.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(ABC):
    """Interface for turning a batch of texts into L2-normalized vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the vectors this embedder produces."""

    @abstractmethod
    def fit(self, texts: list[str]) -> "Embedder":
        """Fit the embedder on a corpus of texts. Returns self."""

    @abstractmethod
    def transform(self, texts: list[str]) -> np.ndarray:
        """Embed texts into an (n_texts, dim) float32 array of unit vectors."""

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist fitted state to path."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "Embedder":
        """Load fitted state from path, previously written by save()."""


class TfidfSvdEmbedder(Embedder):
    """TF-IDF vectorization followed by truncated SVD (LSA) to a fixed dim.

    TF-IDF alone produces high-dimensional sparse vectors; SVD projects them
    down to a small dense space so they can live in a FAISS flat index and so
    unrelated corpora produce comparably-sized indexes. Output vectors are
    L2-normalized so inner product in the vector store is equivalent to
    cosine similarity.
    """

    def __init__(self, n_components: int = 128, random_state: int = 42):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.requested_n_components = n_components
        self.random_state = random_state
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    @property
    def dim(self) -> int:
        if self._svd is None:
            raise RuntimeError("Embedder has not been fit yet")
        return self._svd.n_components

    def fit(self, texts: list[str]) -> "TfidfSvdEmbedder":
        if not texts:
            raise ValueError("Cannot fit an embedder on zero texts")

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf_matrix = self._vectorizer.fit_transform(texts)

        n_features = tfidf_matrix.shape[1]
        n_samples = tfidf_matrix.shape[0]
        # TruncatedSVD requires n_components < n_features, and a component
        # count near n_samples is meaningless (nothing left for SVD to find
        # structure across), so clamp to whichever bound is tighter. This
        # matters for small/test corpora where the requested 128 components
        # would otherwise raise inside scikit-learn.
        safe_n_components = max(1, min(self.requested_n_components, n_features - 1, n_samples - 1))
        if safe_n_components < 1:
            safe_n_components = 1

        self._svd = TruncatedSVD(
            n_components=safe_n_components, random_state=self.random_state
        )
        self._svd.fit(tfidf_matrix)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("Embedder has not been fit yet; call fit() first")
        tfidf_matrix = self._vectorizer.transform(texts)
        dense = self._svd.transform(tfidf_matrix)
        normalized = normalize(dense, norm="l2", axis=1)
        return normalized.astype(np.float32)

    def save(self, path: Path) -> None:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("Cannot save an embedder that has not been fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "requested_n_components": self.requested_n_components,
                    "random_state": self.random_state,
                    "vectorizer": self._vectorizer,
                    "svd": self._svd,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "TfidfSvdEmbedder":
        with open(path, "rb") as f:
            state = pickle.load(f)
        embedder = cls(
            n_components=state["requested_n_components"],
            random_state=state["random_state"],
        )
        embedder._vectorizer = state["vectorizer"]
        embedder._svd = state["svd"]
        return embedder
