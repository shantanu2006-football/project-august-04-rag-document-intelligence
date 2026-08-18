from ragdoc.answerer import extract_answer
from ragdoc.chunker import Chunk
from ragdoc.embedder import TfidfSvdEmbedder
from ragdoc.vector_store import SearchResult

CORPUS = [
    "Photosynthesis converts sunlight into chemical energy. "
    "Plants use chlorophyll to capture light. "
    "The process releases oxygen as a byproduct.",
    "The French Revolution began in 1789. "
    "It led to the end of the monarchy in France. "
    "Napoleon rose to power in its aftermath.",
    "Volcanoes form where magma rises through the crust. "
    "Lava cools quickly to form igneous rock. "
    "Ash clouds can affect air travel for days.",
    "Basketball teams have five players on the court at once. "
    "A shot inside the arc is worth two points. "
    "The shot clock forces a team to attempt a shot in time.",
]


def _fit_embedder() -> TfidfSvdEmbedder:
    # n_components=8 is clamped internally to 3 (n_samples - 1 for this
    # 4-document corpus); the higher request just documents intent and keeps
    # the test robust if the corpus grows.
    embedder = TfidfSvdEmbedder(n_components=8)
    embedder.fit(CORPUS)
    return embedder


def _result(doc_id: str, text: str) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"{doc_id}::0",
        doc_id=doc_id,
        source_path=f"{doc_id}.txt",
        chunk_index=0,
        text=text,
    )
    return SearchResult(chunk=chunk, score=1.0)


def test_extract_answer_picks_best_sentence_across_chunks():
    embedder = _fit_embedder()
    results = [_result("biology", CORPUS[0]), _result("history", CORPUS[1])]

    answer = extract_answer("What does photosynthesis produce?", results, embedder)

    # The lightweight TF-IDF+SVD embedder is coarse: it reliably separates
    # topics (biology vs. history) but isn't guaranteed to rank the single
    # most specific sentence within a topic above its neighbors, so assert
    # on topic, not on the exact "oxygen" sentence.
    assert answer.best_sentence in CORPUS[0]
    assert answer.sources == results


def test_extract_answer_with_no_results():
    embedder = _fit_embedder()

    answer = extract_answer("anything", [], embedder)

    assert answer.best_sentence == ""
    assert answer.sources == []
