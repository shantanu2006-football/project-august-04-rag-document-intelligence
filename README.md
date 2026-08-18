# RAG Document Intelligence

A retrieval-augmented Q&A system over a small corpus of text and PDF documents,
with a hand-labeled evaluation harness that measures retrieval precision,
recall, and MRR — not just a demo that "looks like it works."

## Problem statement

Given a corpus of documents (here: six short technical articles, five as
`.txt` and one as a generated `.pdf`), answer natural-language questions by
retrieving the passages most relevant to the question rather than relying on
a model's memorized knowledge. The system should:

1. Ingest heterogeneous source files (`.txt`, `.md`, `.pdf`) into a searchable
   index.
2. Given a question, retrieve the top-k most relevant passages.
3. Produce a short extractive answer grounded in that retrieved context.
4. Be evaluated against a hand-labeled question set with real, measurable
   retrieval metrics (precision@k, recall@k, MRR, hit rate) rather than
   eyeballed output — retrieval quality is the primary lever for RAG answer
   quality, so it's the primary thing this repo measures.

## Architecture

```
data/corpus/*.{txt,pdf}
        │
        ▼
   loaders.py    ── extract raw text (pypdf for PDFs)
        │
        ▼
   chunker.py    ── sentence-aware, word-budgeted chunks with overlap
        │
        ▼
   embedder.py   ── TF-IDF → truncated SVD → L2-normalized dense vectors
        │
        ▼
  vector_store.py ── FAISS IndexFlatIP (cosine via normalized inner product)
        │
        ├── retrieve(question, k) ──► top-k chunks + scores
        │
        ▼
   answerer.py   ── best-matching sentence across retrieved chunks
        │
        ▼
  evaluation.py  ── precision@k / recall@k / MRR / hit-rate vs. data/eval/questions.json
```

`pipeline.py` (`RagPipeline`) wires these stages together and is the only
class most callers need: `RagPipeline.build(corpus_dir)` to ingest,
`.retrieve()` / `.answer()` to query, `.save()` / `.load()` to persist an
index to disk. `cli.py` exposes `ingest`, `query`, and `evaluate` as
subcommands.

### Design decisions

**TF-IDF + truncated SVD (LSA) instead of a neural sentence embedder.**
A neural embedder (e.g. a sentence-transformer) would likely retrieve
somewhat better on paraphrased or purely semantic queries, but it requires
downloading and running a ~100MB+ model. For a project this size, that
trades away determinism, offline reproducibility, and CI speed (this repo's
whole test suite, including embedding fits, runs in under two seconds) for a
retrieval-quality gain that a classic TF-IDF+LSA embedder already mostly
captures on a small, single-topic-per-document corpus like this one — see
"Future Work" for how to swap it in without touching the rest of the
pipeline, since `Embedder` is an explicit interface.

**FAISS flat (brute-force) index instead of an approximate index.**
`IndexFlatIP` does exact search. For tens to low thousands of chunks — the
realistic range for a "small corpus" system — brute-force search is already
sub-millisecond, so trading accuracy for the speed of an approximate index
like HNSW or IVF-PQ isn't worth the complexity yet. `vector_store.py`
isolates all FAISS-specific code behind a small `VectorStore` class, so
swapping the index type later is a one-file change.

**Document-level ground truth for evaluation, not chunk-level.**
`data/eval/questions.json` labels each question with the *document(s)* it
should be answered from, not exact chunk IDs. Chunk-level labels would be
more precise, but they're brittle: re-running ingestion with different
chunking parameters (`--max-words`, `--overlap-sentences`) shifts chunk
boundaries and IDs, silently invalidating old labels. Because every sample
document in this corpus covers one distinct topic, document-level relevance
is still a meaningful, non-trivial signal — a retrieval that surfaces the
wrong document scores zero regardless of how it was chunked.

**Extractive answering, not generative.** `answerer.py` picks the single
sentence (across the retrieved chunks) most similar to the question and
returns it alongside the full retrieved context. This is a real, tested
answer-selection step, not a stub — but it is not an LLM writing a fluent
answer. See "Future Work."

**Sentence-aware chunking with overlap.** `chunker.py` splits on sentence
boundaries (not fixed character windows) and packs sentences into chunks up
to a word budget, carrying the last `--overlap-sentences` sentences into the
next chunk. This avoids severing a sentence mid-thought at a chunk boundary,
and the overlap keeps a fact from being stranded without the sentence that
explains it.

## Setup

Requires Python 3.11+ (numpy 2.4.6, the pinned version, has no wheels for 3.10).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# 1. Ingest the sample corpus into a local FAISS index
python -m ragdoc.cli ingest
# Indexed 30 chunks from 'data/corpus' into '.ragdoc_index' (embedding dim=29)

# 2. Ask a question against it
python -m ragdoc.cli query "What does self-attention fix that RNNs struggled with?"
```

Example output:

```
Q: What does self-attention fix that RNNs struggled with?

A: The Transformer, introduced in the paper "Attention Is All You Need", replaced recurrence entirely with a mechanism called self-attention.

Top 5 retrieved chunks:
  [1] score=0.770 chunk=transformer_architecture::0 :: The Transformer Architecture and Self-Attention Before 2017, the dominant architectures for sequence modeling were recurrent neural networks (RNNs) and their ga...
  [2] score=0.694 chunk=transformer_architecture::1 :: The Transformer, introduced in the paper "Attention Is All You Need", replaced recurrence entirely with a mechanism called self-attention. In self-attention, ev...
  [3] score=0.546 chunk=transformer_architecture::2 :: Because every position attends to every other position directly, the "distance" between any two tokens in the computation graph is a single step, which fixes th...
```

```bash
# 3. Run the retrieval evaluation harness
python -m ragdoc.cli evaluate
```

Example output (14 hand-labeled questions in `data/eval/questions.json`, k=5):

```
Retrieval evaluation (k=5, n=14 questions)
------------------------------------------------------------
  Mean Precision@5:  0.814
  Mean Recall@5:     1.000
  Mean Reciprocal Rank:      1.000
  Hit Rate@5:        1.000
------------------------------------------------------------
Per-question detail:
  [HIT ] q1: P=0.60 R=1.00 RR=1.00 expected=['distributed_systems'] got=['distributed_systems', 'distributed_systems', 'distributed_systems', 'software_testing', 'software_testing']
  [HIT ] q3: P=1.00 R=1.00 RR=1.00 expected=['vector_databases'] got=['vector_databases', 'vector_databases', 'vector_databases', 'vector_databases', 'vector_databases']
  ...
```

All 14 questions hit (at least one relevant document in the top 5) with a
perfect mean reciprocal rank — the relevant document is always the top
result, and recall@5 is 1.0 across the board. Precision@5 is below 1.0 for
two distinct reasons: some single-topic documents (e.g.
`python_packaging`, `transformer_architecture`) only produce 4 chunks total,
so no retrieval can fill all 5 slots with relevant chunks regardless of
quality; on other questions (e.g. q1) a genuinely unrelated chunk from
another document ranks into the top 5. `python -m ragdoc.cli evaluate` with
a smaller `--k` shows this: at k=3, mean precision rises to 0.905 with the
same perfect recall and MRR.

All CLI flags (`--corpus-dir`, `--index-dir`, `--max-words`,
`--overlap-sentences`, `--n-components`, `--k`, `--eval-file`) are visible via
`python -m ragdoc.cli <command> --help`.

## Running tests

```bash
pytest tests/ -v
```

47 tests cover the loader (text/PDF extraction, unsupported types, empty
files), chunker (sentence splitting, word budgets, overlap, edge cases),
embedder (shape, normalization, fit/save/load round-trips, small-corpus
clamping), vector store (search correctness, save/load, error handling),
pipeline (end-to-end build/retrieve/answer/persist), evaluation metrics
(precision/recall/MRR arithmetic on controlled inputs), and the CLI
(`ingest`/`query`/`evaluate` subcommands end to end).

## Project layout

```
src/ragdoc/          # the package: loaders, chunker, embedder, vector_store,
                      # answerer, pipeline, evaluation, cli
tests/                # pytest unit + integration tests, one file per module
data/corpus/          # 6 sample documents (5 .txt + 1 generated .pdf)
data/eval/            # hand-labeled question set for retrieval evaluation
scripts/               # one-off content-authoring script (generates the sample PDF)
.github/workflows/    # CI: install + test on Python 3.10/3.11/3.12
```

## Future work

Cut from this session's scope to keep it tight and correct rather than broad
and shaky:

- **Neural embeddings.** Swap `TfidfSvdEmbedder` for a sentence-transformer
  behind the existing `Embedder` interface, and A/B the two on the same
  evaluation harness — the harness doesn't change, only the embedder does.
- **Generative answers.** Replace/augment `answerer.py`'s extractive
  best-sentence selection with a call to an LLM (e.g. the Claude API) that
  synthesizes an answer from the retrieved chunks, with citations back to
  `chunk_id`.
- **Approximate indexing at scale.** If the corpus grows past low thousands
  of chunks, swap `IndexFlatIP` for `IndexHNSWFlat` or `IndexIVFPQ` inside
  `vector_store.py`.
- **Chunk-level relevance judgments** for finer-grained retrieval metrics,
  once chunking parameters are stable enough that hand labels won't rot.
- **Hybrid retrieval** (BM25 + dense, reciprocal rank fusion) to combine
  exact keyword matches with semantic similarity.
