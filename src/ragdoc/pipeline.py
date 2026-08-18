"""Ties loaders, chunker, embedder, and vector store into one RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from ragdoc.answerer import Answer, extract_answer
from ragdoc.chunker import chunk_corpus
from ragdoc.embedder import Embedder, TfidfSvdEmbedder
from ragdoc.loaders import load_corpus
from ragdoc.vector_store import SearchResult, VectorStore

_EMBEDDER_FILENAME = "embedder.pkl"


class RagPipeline:
    """Ingests a document corpus and answers questions against it."""

    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    @classmethod
    def build(
        cls,
        corpus_dir: Path,
        max_words: int = 120,
        overlap_sentences: int = 1,
        n_components: int = 128,
    ) -> "RagPipeline":
        """Load, chunk, embed, and index every document in corpus_dir."""
        documents = load_corpus(corpus_dir)
        chunks = chunk_corpus(
            documents, max_words=max_words, overlap_sentences=overlap_sentences
        )
        if not chunks:
            raise ValueError(f"Corpus at {corpus_dir} produced zero chunks")

        embedder = TfidfSvdEmbedder(n_components=n_components)
        vectors = embedder.fit_transform([c.text for c in chunks])

        vector_store = VectorStore(dim=embedder.dim)
        vector_store.add(vectors, chunks)

        return cls(embedder=embedder, vector_store=vector_store)

    def retrieve(self, question: str, k: int = 5) -> list[SearchResult]:
        """Return the top-k chunks most relevant to question, best first."""
        query_vector = self.embedder.transform([question])[0]
        return self.vector_store.search(query_vector, k=k)

    def answer(self, question: str, k: int = 5) -> Answer:
        """Retrieve context for question and extract a best-sentence answer."""
        results = self.retrieve(question, k=k)
        return extract_answer(question, results, self.embedder)

    def save(self, index_dir: Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        self.embedder.save(index_dir / _EMBEDDER_FILENAME)
        self.vector_store.save(index_dir)

    @classmethod
    def load(cls, index_dir: Path) -> "RagPipeline":
        index_dir = Path(index_dir)
        embedder = TfidfSvdEmbedder.load(index_dir / _EMBEDDER_FILENAME)
        vector_store = VectorStore.load(index_dir)
        return cls(embedder=embedder, vector_store=vector_store)
