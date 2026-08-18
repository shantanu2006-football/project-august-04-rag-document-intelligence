"""A FAISS-backed store of chunk vectors with their source metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from ragdoc.chunker import Chunk

_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "chunks.json"


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with its similarity score for a query."""

    chunk: Chunk
    score: float


class VectorStore:
    """Wraps a FAISS flat inner-product index plus parallel chunk metadata.

    Vectors are expected to already be L2-normalized (see embedder.py), which
    makes inner product equivalent to cosine similarity. A flat (brute-force)
    index is deliberately used instead of an approximate index like HNSW or
    IVF-PQ: for a corpus of this size (tens to low thousands of chunks) exact
    search is fast enough that trading accuracy for speed isn't worth it. See
    README "Design Decisions" for the trade-off at larger scale.
    """

    def __init__(self, dim: int):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Add vectors and their corresponding chunks to the store.

        vectors[i] must correspond to chunks[i]. Vectors are appended in
        order, so FAISS's internal sequential id for a chunk is stable and
        equal to its index within self._chunks.
        """
        if vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Got {vectors.shape[0]} vectors but {len(chunks)} chunks; "
                "they must be the same length and index-aligned"
            )
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Vector dim {vectors.shape[1]} does not match store dim {self.dim}"
            )
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Return the top-k chunks most similar to query_vector, best first."""
        if len(self) == 0:
            return []
        if k < 1:
            raise ValueError("k must be >= 1")
        query = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {query.shape[1]} does not match store dim {self.dim}"
            )
        effective_k = min(k, len(self))
        scores, indices = self._index.search(query, effective_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(SearchResult(chunk=self._chunks[idx], score=float(score)))
        return results

    def save(self, dir_path: Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(dir_path / _INDEX_FILENAME))
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "text": c.text,
            }
            for c in self._chunks
        ]
        with open(dir_path / _METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(chunk_dicts, f, indent=2)

    @classmethod
    def load(cls, dir_path: Path) -> "VectorStore":
        dir_path = Path(dir_path)
        index = faiss.read_index(str(dir_path / _INDEX_FILENAME))
        with open(dir_path / _METADATA_FILENAME, encoding="utf-8") as f:
            chunk_dicts = json.load(f)
        store = cls(dim=index.d)
        store._index = index
        store._chunks = [Chunk(**d) for d in chunk_dicts]
        return store
