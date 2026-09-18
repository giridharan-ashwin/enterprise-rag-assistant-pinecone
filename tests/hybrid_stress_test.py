from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.clients import pinecone_client
from app.config import settings

from tests.hybrid_eval import dense_retrieve
from tests.stress_test import (
    expected_sections,
    expected_sources,
    get_question,
    is_unknown_case,
    load_cases,
)
from app.services.hybrid import (
    bm25_retriever,
    reciprocal_rank_fusion,
)


NAMESPACE = "stress-test"

# Dataset-specific threshold selected from the earlier sweep.
HYBRID_DENSE_THRESHOLD = 0.50

DENSE_TOP_K = 10
SPARSE_TOP_K = 10
FINAL_TOP_K = 5

RERANK_MIN_INTERVAL = 1.1
_last_rerank_time = 0.0

OUTPUT_FILE = Path("results/hybrid_stress_results.json")


def rate_limit_rerank() -> None:
    global _last_rerank_time

    now = time.monotonic()
    elapsed = now - _last_rerank_time

    if elapsed < RERANK_MIN_INTERVAL:
        time.sleep(RERANK_MIN_INTERVAL - elapsed)

    _last_rerank_time = time.monotonic()


def hybrid_candidates(
    question: str,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Run dense + BM25 and combine with Reciprocal Rank Fusion.

    When threshold is supplied, it acts as a dense answerability gate
    and filters dense candidates before fusion.
    """

    dense_results = dense_retrieve(
        question,
        top_k=DENSE_TOP_K,
    )

    sparse_results = bm25_retriever.retrieve(
        question,
        top_k=SPARSE_TOP_K,
    )

    if threshold is not None:
        dense_results = [
            result
            for result in dense_results
            if float(result["score"]) >= threshold
        ]

        # Dense threshold acts as an abstention gate.
        # If there is no sufficiently relevant dense evidence,
        # reject instead of allowing BM25 alone to manufacture a result.
        if not dense_results:
            return []

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=FINAL_TOP_K,
    )


def rerank(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    documents = [
        {
            "id": str(index),
            "text": candidate["text"],
        }
        for index, candidate in enumerate(candidates)
    ]

    rate_limit_rerank()

    response = pinecone_client.inference.rerank(
        model=settings.rerank_model,
        query=question,
        documents=documents,
        top_n=min(settings.rerank_top_n, len(documents)),
        return_documents=False,
    )

    results = []

    for item in response.data:
        candidate = candidates[item.index]

        results.append(
            {
                **candidate,
                "rerank_score": float(item.score),
            }
        )

    return results


def retrieve_for_mode(
    question: str,
    mode: str,
) -> list[dict[str, Any]]:

    if mode == "hybrid":
        return hybrid_candidates(
            question,
            threshold=None,
        )

    if mode == "hybrid_threshold":
        return hybrid_candidates(
            question,
            threshold=HYBRID_DENSE_THRESHOLD,
        )

    if mode == "hybrid_threshold_rerank":
        candidates = hybrid_candidates(
            question,
            threshold=HYBRID_DENSE_THRESHOLD,
        )

        return rerank(
            question,
            candidates,
        )

    raise ValueError(f"Unknown mode: {mode}")


def evaluate_case(
    case: dict[str, Any],
    mode: str,
    index: int,
) -> dict[str, Any]:

    question = get_question(case)
    results = retrieve_for_mode(question, mode)

    unknown = is_unknown_case(case)

    expected_source_set = expected_sources(case)
    expected_section_set = expected_sections(case)

    if unknown:
        top1_correct = len(results) == 0
        recall_at_3 = len(results) == 0
    else:
        if not results:
            top1_correct = False
            recall_at_3 = False
        else:
            top = results[0]

            top_source = str(
                top.get("source", "")
            ).strip().lower()

            top_section = str(
                top.get("section", "")
            ).strip().lower()

            source_top_match = (
                not expected_source_set
                or top_source in expected_source_set
            )

            section_top_match = (
                not expected_section_set
                or top_section in expected_section_set
            )

            top1_correct = (
                source_top_match
                and section_top_match
            )

            returned_sources = {
                str(result.get("source", ""))
                .strip()
                .lower()
                for result in results[:3]
            }

            returned_sections = {
                str(result.get("section", ""))
                .strip()
                .lower()
                for result in results[:3]
            }

            source_recall_match = (
                not expected_source_set
                or bool(returned_sources & expected_source_set)
            )

            section_recall_match = (
                not expected_section_set
                or bool(returned_sections & expected_section_set)
            )

            recall_at_3 = (
                source_recall_match
                and section_recall_match
            )

    return {
        "id": str(
            case.get("id")
            or case.get("question_id")
            or f"Q{index + 1}"
        ),
        "question": question,
        "unknown": unknown,
        "top1_correct": top1_correct,
        "recall_at_3": recall_at_3,
        "results": results,
    }


def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:

    known_results = [
        result
        for result in results
        if not result["unknown"]
    ]

    unknown_results = [
        result
        for result in results
        if result["unknown"]
    ]

    known_top1 = (
        sum(result["top1_correct"] for result in known_results)
        / len(known_results)
        if known_results
        else 0.0
    )

    known_recall_at_3 = (
        sum(result["recall_at_3"] for result in known_results)
        / len(known_results)
        if known_results
        else 0.0
    )

    unknown_rejection = (
        sum(result["recall_at_3"] for result in unknown_results)
        / len(unknown_results)
        if unknown_results
        else 0.0
    )

    known_rejection = (
        sum(
            1
            for result in known_results
            if not result["results"]
        )
        / len(known_results)
        if known_results
        else 0.0
    )

    total = len(results)

    overall_accuracy = (
        (
            sum(result["top1_correct"] for result in known_results)
            + sum(result["recall_at_3"] for result in unknown_results)
        )
        / total
        if total
        else 0.0
    )

    return {
        "known_top1": known_top1,
        "known_recall_at_3": known_recall_at_3,
        "unknown_rejection": unknown_rejection,
        "known_rejection": known_rejection,
        "overall_accuracy": overall_accuracy,
    }


def print_failures(
    results: list[dict[str, Any]],
    mode: str,
) -> None:

    failures = [
        result
        for result in results
        if not result["top1_correct"]
    ]

    if not failures:
        print("\nNo Top-1 failures.")
        return

    print()
    print(f"--- {mode.upper()} FAILURES ---")

    for result in failures:
        print()
        print(f"[{result['id']}] {result['question']}")
        print(f"Expected unknown: {result['unknown']}")

        if not result["results"]:
            print("Retrieved: NONE")
            continue

        for rank, retrieved in enumerate(
            result["results"],
            start=1,
        ):
            dense_score = retrieved.get("dense_score")
            bm25_score = retrieved.get("bm25_score")
            rrf_score = retrieved.get("rrf_score")
            rerank_score = retrieved.get("rerank_score")

            details = []

            if dense_score is not None:
                details.append(
                    f"dense={float(dense_score):.4f}"
                )

            if bm25_score is not None:
                details.append(
                    f"bm25={float(bm25_score):.4f}"
                )

            if rrf_score is not None:
                details.append(
                    f"rrf={float(rrf_score):.5f}"
                )

            if rerank_score is not None:
                details.append(
                    f"rerank={float(rerank_score):.4f}"
                )

            print(
                f"  {rank}. "
                f"{retrieved.get('source')} | "
                f"{retrieved.get('section')} | "
                f"{' | '.join(details)}"
            )


def save_results(
    all_results: dict[str, list[dict[str, Any]]],
) -> None:

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "namespace": NAMESPACE,
        "hybrid_dense_threshold": HYBRID_DENSE_THRESHOLD,
        "reranker": settings.rerank_model,
        "results": all_results,
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:

    cases = load_cases()

    print()
    print("=" * 100)
    print("ENTERPRISE RAG HYBRID STRESS TEST")
    print("=" * 100)

    print(f"Cases:                  {len(cases)}")
    print(f"Namespace:              {NAMESPACE}")
    print(
        f"Hybrid dense threshold: {HYBRID_DENSE_THRESHOLD}"
    )
    print(
        f"Reranker:               {settings.rerank_model}"
    )
    print("=" * 100)

    modes = [
        ("hybrid", "Hybrid"),
        (
            "hybrid_threshold",
            "Hybrid + Threshold",
        ),
        (
            "hybrid_threshold_rerank",
            "Hybrid + Threshold + Reranking",
        ),
    ]

    all_results = {}
    metrics_table = {}

    for mode, display_name in modes:

        print()
        print("=" * 100)
        print(f"RUNNING: {display_name}")
        print("=" * 100)

        mode_results = []

        for index, case in enumerate(cases):

            result = evaluate_case(
                case,
                mode,
                index,
            )

            mode_results.append(result)

            if (
                mode == "hybrid_threshold_rerank"
                and (index + 1) % 25 == 0
            ):
                print(
                    f"Reranked {index + 1}/{len(cases)} cases..."
                )

        all_results[display_name] = mode_results

        metrics = calculate_metrics(
            mode_results
        )

        metrics_table[display_name] = metrics

        print()
        print(
            f"Known Top-1:       "
            f"{metrics['known_top1'] * 100:.2f}%"
        )
        print(
            f"Known Recall@3:   "
            f"{metrics['known_recall_at_3'] * 100:.2f}%"
        )
        print(
            f"Unknown Rejection:"
            f" {metrics['unknown_rejection'] * 100:.2f}%"
        )
        print(
            f"Known Rejection:   "
            f"{metrics['known_rejection'] * 100:.2f}%"
        )
        print(
            f"Overall Accuracy:  "
            f"{metrics['overall_accuracy'] * 100:.2f}%"
        )

        print_failures(
            mode_results,
            mode,
        )

    save_results(all_results)

    print()
    print("=" * 110)
    print("HYBRID STRESS TEST SUMMARY")
    print("=" * 110)

    print(
        f"{'Configuration':<36}"
        f"{'Known Top-1':>14}"
        f"{'Recall@3':>14}"
        f"{'Unknown Reject':>18}"
        f"{'Known Reject':>16}"
        f"{'Overall':>14}"
    )

    print("-" * 110)

    for display_name, metrics in metrics_table.items():

        print(
            f"{display_name:<36}"
            f"{metrics['known_top1'] * 100:>13.2f}%"
            f"{metrics['known_recall_at_3'] * 100:>13.2f}%"
            f"{metrics['unknown_rejection'] * 100:>17.2f}%"
            f"{metrics['known_rejection'] * 100:>15.2f}%"
            f"{metrics['overall_accuracy'] * 100:>13.2f}%"
        )

    print("=" * 110)

    print()
    print(
        f"Detailed results saved to: "
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
