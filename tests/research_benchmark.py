from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from app.clients import pinecone_client
from app.config import settings
from app.services.hybrid import reciprocal_rank_fusion

from tests.eval_utils import (
    get_question,
    is_unknown_case,
    is_multi_section_case,
    expected_targets,
)
from tests.stress_test import load_cases


NAMESPACE = "stress-test"

# Dataset-specific experimental operating point
THRESHOLD = 0.50

DENSE_CANDIDATES = 5
HYBRID_FINAL_TOP_K = 5

CACHE_FILE = Path(
    "results/retrieval_cache.json"
)

RERANK_CACHE_FILE = Path(
    "results/rerank_cache.json"
)

OUTPUT_FILE = Path(
    "results/research_benchmark.json"
)

RERANK_MIN_INTERVAL = 1.25

_last_rerank_time = 0.0


# -------------------------------------------------------------------
# File helpers
# -------------------------------------------------------------------

def load_json_file(
    path: Path,
    default: dict | None = None,
) -> dict:

    if not path.exists():
        return default or {}

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json_file(
    path: Path,
    data: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


# -------------------------------------------------------------------
# Reranking
# -------------------------------------------------------------------

def wait_for_rerank_slot() -> None:

    global _last_rerank_time

    elapsed = (
        time.monotonic()
        - _last_rerank_time
    )

    if elapsed < RERANK_MIN_INTERVAL:
        time.sleep(
            RERANK_MIN_INTERVAL
            - elapsed
        )


def rerank_with_cache(
    case_id: str,
    question: str,
    candidates: list[dict[str, Any]],
    cache: dict[str, Any],
) -> list[dict[str, Any]]:

    if not candidates:
        return []

    cache_key = case_id

    if cache_key in cache:
        return cache[cache_key]

    global _last_rerank_time

    documents = [
        {
            "id": str(index),
            "text": candidate["text"],
        }
        for index, candidate
        in enumerate(candidates)
    ]

    max_attempts = 6

    for attempt in range(max_attempts):

        wait_for_rerank_slot()

        try:

            response = (
                pinecone_client
                .inference
                .rerank(
                    model=settings.rerank_model,
                    query=question,
                    documents=documents,
                    top_n=min(
                        settings.rerank_top_n,
                        len(documents),
                    ),
                    return_documents=False,
                )
            )

            _last_rerank_time = time.monotonic()

            results = []

            for item in response.data:

                candidate = candidates[
                    item.index
                ]

                results.append(
                    {
                        **candidate,
                        "rerank_score": float(
                            item.score
                        ),
                    }
                )

            cache[cache_key] = results

            # Persist after every successful request.
            save_json_file(
                RERANK_CACHE_FILE,
                cache,
            )

            return results

        except Exception as exc:

            message = str(exc)

            is_rate_limit = (
                "429" in message
                or "RESOURCE_EXHAUSTED"
                in message
                or "Too Many Requests"
                in message
            )

            if not is_rate_limit:
                raise

            wait_seconds = min(
                10 * (attempt + 1),
                60,
            )

            print(
                f"Reranker rate limit. "
                f"Waiting {wait_seconds}s "
                f"before retry "
                f"{attempt + 1}/{max_attempts}..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Reranker rate limit persisted "
        "after multiple retries."
    )


# -------------------------------------------------------------------
# Retrieval modes
# -------------------------------------------------------------------

def dense(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    return record["dense"][
        :DENSE_CANDIDATES
    ][:3]


def dense_threshold(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    candidates = [
        result
        for result in record["dense"][
            :DENSE_CANDIDATES
        ]
        if float(result["score"])
        >= THRESHOLD
    ]

    return candidates[:3]


def dense_threshold_rerank(
    case_id: str,
    record: dict[str, Any],
    rerank_cache: dict[str, Any],
) -> list[dict[str, Any]]:

    candidates = [
        result
        for result in record["dense"][
            :DENSE_CANDIDATES
        ]
        if float(result["score"])
        >= THRESHOLD
    ]

    reranked = rerank_with_cache(
        case_id=f"dense:{case_id}",
        question=record["question"],
        candidates=candidates,
        cache=rerank_cache,
    )

    return reranked[:3]


def hybrid(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    return reciprocal_rank_fusion(
        dense_results=record["dense"],
        sparse_results=record["sparse"],
        top_k=HYBRID_FINAL_TOP_K,
    )


def hybrid_threshold(
    record: dict[str, Any],
) -> list[dict[str, Any]]:

    dense_candidates = [
        result
        for result in record["dense"]
        if float(result["score"])
        >= THRESHOLD
    ]

    if not dense_candidates:
        return []

    return reciprocal_rank_fusion(
        dense_results=dense_candidates,
        sparse_results=record["sparse"],
        top_k=HYBRID_FINAL_TOP_K,
    )


def hybrid_threshold_rerank(
    case_id: str,
    record: dict[str, Any],
    rerank_cache: dict[str, Any],
) -> list[dict[str, Any]]:

    candidates = hybrid_threshold(record)

    reranked = rerank_with_cache(
        case_id=f"hybrid:{case_id}",
        question=record["question"],
        candidates=candidates,
        cache=rerank_cache,
    )

    return reranked[:3]


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def target_match(
    result: dict[str, Any],
    target: tuple[str, str],
) -> bool:

    source, section = target

    actual_source = str(
        result.get("source", "")
    ).strip().lower()

    actual_section = str(
        result.get("section", "")
    ).strip().lower()

    source_match = (
        not source
        or actual_source == source
    )

    section_match = (
        not section
        or actual_section == section
    )

    return (
        source_match
        and section_match
    )


def evaluate_result(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    unknown = is_unknown_case(case)
    multi = is_multi_section_case(case)

    targets = expected_targets(case)

    if unknown:

        rejected = (
            len(results) == 0
        )

        return {
            "top1": rejected,
            "recall_at_3": rejected,
            "multi_section_recall": None,
            "results": results,
        }

    if not results:

        return {
            "top1": False,
            "recall_at_3": False,
            "multi_section_recall": (
                0.0
                if multi
                else None
            ),
            "results": [],
        }

    # Top-1:
    # For multi-section cases the first result only needs
    # to be one of the required targets.
    top1 = any(
        target_match(
            results[0],
            target,
        )
        for target in targets
    )

    top3 = results[:3]

    if multi and targets:

        retrieved_pairs = {
            (
                str(
                    result.get(
                        "source",
                        "",
                    )
                ).strip().lower(),
                str(
                    result.get(
                        "section",
                        "",
                    )
                ).strip().lower(),
            )
            for result in top3
        }

        recall_at_3 = all(
            target in retrieved_pairs
            for target in targets
        )

    else:

        recall_at_3 = any(
            target_match(
                result,
                target,
            )
            for result in top3
            for target in targets
        )

    if multi and targets:

        retrieved_pairs = {
            (
                str(
                    result.get(
                        "source",
                        "",
                    )
                ).strip().lower(),
                str(
                    result.get(
                        "section",
                        "",
                    )
                ).strip().lower(),
            )
            for result in results[:5]
        }

        hits = sum(
            target in retrieved_pairs
            for target in targets
        )

        multi_recall = (
            hits / len(targets)
        )

    else:

        multi_recall = None

    return {
        "top1": top1,
        "recall_at_3": recall_at_3,
        "multi_section_recall": multi_recall,
        "results": results,
    }


def calculate_metrics(
    evaluations: list[dict[str, Any]],
) -> dict[str, float]:

    known = [
        item
        for item in evaluations
        if not item["unknown"]
    ]

    unknown = [
        item
        for item in evaluations
        if item["unknown"]
    ]

    multi = [
        item
        for item in evaluations
        if item["multi_section_recall"]
        is not None
    ]

    known_top1 = (
        sum(
            item["top1"]
            for item in known
        )
        / len(known)
        if known
        else 0.0
    )

    known_recall = (
        sum(
            item["recall_at_3"]
            for item in known
        )
        / len(known)
        if known
        else 0.0
    )

    unknown_rejection = (
        sum(
            item["recall_at_3"]
            for item in unknown
        )
        / len(unknown)
        if unknown
        else 0.0
    )

    known_rejection = (
        sum(
            not item["results"]
            for item in known
        )
        / len(known)
        if known
        else 0.0
    )

    overall = (
        (
            sum(
                item["top1"]
                for item in known
            )
            +
            sum(
                item["recall_at_3"]
                for item in unknown
            )
        )
        / len(evaluations)
        if evaluations
        else 0.0
    )

    multi_recall = (
        sum(
            item["multi_section_recall"]
            for item in multi
        )
        / len(multi)
        if multi
        else 0.0
    )

    return {
        "known_top1": known_top1,
        "known_recall_at_3": known_recall,
        "unknown_rejection": unknown_rejection,
        "known_rejection": known_rejection,
        "overall_accuracy": overall,
        "multi_section_recall": multi_recall,
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:

    cases = load_cases()

    retrieval_cache = load_json_file(
        CACHE_FILE
    )

    if len(retrieval_cache) != len(cases):
        raise RuntimeError(
            "Retrieval cache is missing or incomplete. "
            "Run:\n"
            "python -m tests.build_retrieval_cache"
        )

    rerank_cache = load_json_file(
        RERANK_CACHE_FILE
    )

    print()
    print("=" * 120)
    print("ENTERPRISE RAG 6-WAY RESEARCH BENCHMARK")
    print("=" * 120)

    print(
        f"Cases:              {len(cases)}"
    )

    print(
        f"Namespace:          {NAMESPACE}"
    )

    print(
        f"Threshold:          {THRESHOLD}"
    )

    print(
        f"Reranker:            "
        f"{settings.rerank_model}"
    )

    print(
        f"Cached rerank sets:  "
        f"{len(rerank_cache)}"
    )

    modes = [
        ("dense", "Dense"),
        (
            "dense_threshold",
            "Dense + Threshold",
        ),
        (
            "dense_threshold_rerank",
            "Dense + Threshold + Rerank",
        ),
        ("hybrid", "Hybrid"),
        (
            "hybrid_threshold",
            "Hybrid + Threshold",
        ),
        (
            "hybrid_threshold_rerank",
            "Hybrid + Threshold + Rerank",
        ),
    ]

    summary = {}
    detailed_results = {}

    for mode, display_name in modes:

        print()
        print("=" * 120)
        print(
            f"RUNNING: {display_name}"
        )
        print("=" * 120)

        evaluations = []

        for index, case in enumerate(cases):

            case_id = str(
                case.get("id")
                or case.get("question_id")
                or f"Q{index + 1}"
            )

            record = retrieval_cache[
                case_id
            ]

            if mode == "dense":

                results = dense(record)

            elif mode == "dense_threshold":

                results = dense_threshold(
                    record
                )

            elif mode == "dense_threshold_rerank":

                results = (
                    dense_threshold_rerank(
                        case_id,
                        record,
                        rerank_cache,
                    )
                )

            elif mode == "hybrid":

                results = hybrid(record)

            elif mode == "hybrid_threshold":

                results = hybrid_threshold(
                    record
                )

            elif mode == "hybrid_threshold_rerank":

                results = (
                    hybrid_threshold_rerank(
                        case_id,
                        record,
                        rerank_cache,
                    )
                )

            else:
                raise ValueError(
                    f"Unknown mode: {mode}"
                )

            evaluation = evaluate_result(
                case,
                results,
            )

            evaluation.update(
                {
                    "id": case_id,
                    "question": get_question(case),
                    "unknown": is_unknown_case(case),
                    "multi_section":
                        is_multi_section_case(
                            case
                        ),
                }
            )

            evaluations.append(
                evaluation
            )

        metric = calculate_metrics(
            evaluations
        )

        summary[display_name] = metric
        detailed_results[display_name] = evaluations

        print(
            f"Known Top-1:          "
            f"{metric['known_top1'] * 100:.2f}%"
        )

        print(
            f"Known Recall@3:       "
            f"{metric['known_recall_at_3'] * 100:.2f}%"
        )

        print(
            f"Unknown Rejection:    "
            f"{metric['unknown_rejection'] * 100:.2f}%"
        )

        print(
            f"Known Rejection:      "
            f"{metric['known_rejection'] * 100:.2f}%"
        )

        print(
            f"Overall Accuracy:     "
            f"{metric['overall_accuracy'] * 100:.2f}%"
        )

        print(
            f"Multi-Section Recall: "
            f"{metric['multi_section_recall'] * 100:.2f}%"
        )

    print()
    print("=" * 135)
    print("FINAL 6-WAY RESEARCH COMPARISON")
    print("=" * 135)

    print(
        f"{'Configuration':<38}"
        f"{'Top-1':>12}"
        f"{'Recall@3':>14}"
        f"{'Unknown':>14}"
        f"{'Known Reject':>16}"
        f"{'Overall':>14}"
        f"{'Multi-Sec':>14}"
    )

    print("-" * 135)

    for name, metric in summary.items():

        print(
            f"{name:<38}"
            f"{metric['known_top1'] * 100:>11.2f}%"
            f"{metric['known_recall_at_3'] * 100:>13.2f}%"
            f"{metric['unknown_rejection'] * 100:>13.2f}%"
            f"{metric['known_rejection'] * 100:>15.2f}%"
            f"{metric['overall_accuracy'] * 100:>13.2f}%"
            f"{metric['multi_section_recall'] * 100:>13.2f}%"
        )

    print("=" * 135)

    save_json_file(
        OUTPUT_FILE,
        {
            "namespace": NAMESPACE,
            "threshold": THRESHOLD,
            "reranker": settings.rerank_model,
            "metrics": summary,
            "results": detailed_results,
        },
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()