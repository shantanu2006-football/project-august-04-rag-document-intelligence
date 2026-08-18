"""Command-line interface: ragdoc ingest | query | evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragdoc.evaluation import evaluate_pipeline, format_report, load_question_set
from ragdoc.pipeline import RagPipeline

DEFAULT_INDEX_DIR = Path(".ragdoc_index")
DEFAULT_CORPUS_DIR = Path("data/corpus")
DEFAULT_EVAL_FILE = Path("data/eval/questions.json")


def _cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = RagPipeline.build(
        corpus_dir=args.corpus_dir,
        max_words=args.max_words,
        overlap_sentences=args.overlap_sentences,
        n_components=args.n_components,
    )
    pipeline.save(args.index_dir)
    print(
        f"Indexed {len(pipeline.vector_store)} chunks from '{args.corpus_dir}' "
        f"into '{args.index_dir}' (embedding dim={pipeline.embedder.dim})"
    )


def _cmd_query(args: argparse.Namespace) -> None:
    pipeline = RagPipeline.load(args.index_dir)
    answer = pipeline.answer(args.question, k=args.k)

    print(f"Q: {answer.question}\n")
    print(f"A: {answer.best_sentence or '(no answer found)'}\n")
    print(f"Top {len(answer.sources)} retrieved chunks:")
    for rank, result in enumerate(answer.sources, start=1):
        preview = result.chunk.text[:160].replace("\n", " ")
        print(
            f"  [{rank}] score={result.score:.3f} "
            f"chunk={result.chunk.chunk_id} :: {preview}..."
        )


def _cmd_evaluate(args: argparse.Namespace) -> None:
    pipeline = RagPipeline.load(args.index_dir)
    questions = load_question_set(args.eval_file)
    report = evaluate_pipeline(pipeline, questions, k=args.k)
    print(format_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragdoc", description="Retrieval-augmented document Q&A toolkit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest", help="Chunk, embed, and index a document corpus"
    )
    ingest_parser.add_argument(
        "--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR
    )
    ingest_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    ingest_parser.add_argument("--max-words", type=int, default=120)
    ingest_parser.add_argument("--overlap-sentences", type=int, default=1)
    ingest_parser.add_argument("--n-components", type=int, default=128)
    ingest_parser.set_defaults(func=_cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Ask a question against an index")
    query_parser.add_argument("question", type=str)
    query_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    query_parser.add_argument("--k", type=int, default=5)
    query_parser.set_defaults(func=_cmd_query)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run the retrieval evaluation harness against an index"
    )
    evaluate_parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    evaluate_parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    evaluate_parser.add_argument("--k", type=int, default=5)
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
