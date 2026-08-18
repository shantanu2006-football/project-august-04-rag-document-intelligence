"""Retrieval evaluation harness: precision@k, recall@k, and MRR.

Ground truth is labeled at the document level (data/eval/questions.json maps
each question to the source document(s) it should be answered from) rather
than at the exact chunk level. That is a deliberate scope cut: chunk-level
judgments would be more precise but are brittle to re-running ingestion with
different chunking parameters, since chunk ids and boundaries shift. Because
every sample document in this corpus covers a single, distinct topic,
document-level relevance is still a meaningful and non-trivial signal: a
retrieval that pulls the wrong document scores zero regardless of chunking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragdoc.pipeline import RagPipeline


@dataclass(frozen=True)
class QuestionExample:
    """A hand-labeled (question, relevant documents) pair."""

    id: str
    question: str
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class QuestionResult:
    """Per-question retrieval metrics at a fixed k."""

    id: str
    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class EvalReport:
    """Aggregate metrics over a full question set at a fixed k."""

    k: int
    n_questions: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    hit_rate_at_k: float
    results: list[QuestionResult]


def load_question_set(path: Path) -> list[QuestionExample]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not raw:
        raise ValueError(f"Question set at {path} is empty")
    return [
        QuestionExample(
            id=item["id"],
            question=item["question"],
            relevant_doc_ids=item["relevant_doc_ids"],
        )
        for item in raw
    ]


def _score_question(
    example: QuestionExample, retrieved_doc_ids: list[str], k: int
) -> QuestionResult:
    relevant = set(example.relevant_doc_ids)
    top_k = retrieved_doc_ids[:k]

    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    precision = hits / k if k > 0 else 0.0
    recall = len(set(top_k) & relevant) / len(relevant) if relevant else 0.0

    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return QuestionResult(
        id=example.id,
        question=example.question,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=reciprocal_rank,
        retrieved_doc_ids=top_k,
        relevant_doc_ids=list(relevant),
    )


def evaluate_pipeline(
    pipeline: RagPipeline, questions: list[QuestionExample], k: int = 5
) -> EvalReport:
    """Run retrieval for every question and compute aggregate metrics."""
    if k < 1:
        raise ValueError("k must be >= 1")

    results = []
    for example in questions:
        search_results = pipeline.retrieve(example.question, k=k)
        retrieved_doc_ids = [r.chunk.doc_id for r in search_results]
        results.append(_score_question(example, retrieved_doc_ids, k))

    n = len(results)
    hit_count = sum(1 for r in results if r.reciprocal_rank > 0)
    return EvalReport(
        k=k,
        n_questions=n,
        mean_precision_at_k=sum(r.precision_at_k for r in results) / n,
        mean_recall_at_k=sum(r.recall_at_k for r in results) / n,
        mean_reciprocal_rank=sum(r.reciprocal_rank for r in results) / n,
        hit_rate_at_k=hit_count / n,
        results=results,
    )


def format_report(report: EvalReport) -> str:
    lines = [
        f"Retrieval evaluation (k={report.k}, n={report.n_questions} questions)",
        "-" * 60,
        f"  Mean Precision@{report.k}:  {report.mean_precision_at_k:.3f}",
        f"  Mean Recall@{report.k}:     {report.mean_recall_at_k:.3f}",
        f"  Mean Reciprocal Rank:      {report.mean_reciprocal_rank:.3f}",
        f"  Hit Rate@{report.k}:        {report.hit_rate_at_k:.3f}",
        "-" * 60,
        "Per-question detail:",
    ]
    for r in report.results:
        status = "HIT " if r.reciprocal_rank > 0 else "MISS"
        lines.append(
            f"  [{status}] {r.id}: P={r.precision_at_k:.2f} R={r.recall_at_k:.2f} "
            f"RR={r.reciprocal_rank:.2f} expected={r.relevant_doc_ids} "
            f"got={r.retrieved_doc_ids}"
        )
    return "\n".join(lines)
