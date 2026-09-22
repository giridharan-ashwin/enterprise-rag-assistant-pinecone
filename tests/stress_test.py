from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Any

from app.clients import pinecone_client, pinecone_index
from app.config import settings
from app.services.embeddings import embed_query


CASES_FILE = Path(__file__).with_name("stress_test_cases.json")
NAMESPACE = "stress-test"
SKIP_RERANK = "--skip-rerank" in sys.argv
RERANK_MIN_INTERVAL = 1.1
_last_rerank_time = 0.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def normalize(value: Any) -> str:
    return str(value).strip().lower()


def load_cases() -> list[dict[str, Any]]:
    with CASES_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "cases",
            "questions",
            "test_cases",
            "queries",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Unsupported stress_test_cases.json format"
    )


def get_question(case: dict[str, Any]) -> str:
    for key in (
        "question",
        "query",
        "prompt",
    ):
        value = case.get(key)

        if value:
            return str(value).strip()

    raise ValueError(
        f"No question found in case: {case}"
    )


def case_id(
    case: dict[str, Any],
    index: int,
) -> str:
    return str(
        case.get("id")
        or case.get("question_id")
        or f"Q{index + 1}"
    )


def is_unknown_case(
    case: dict[str, Any],
) -> bool:

    if case.get("known") is False:
        return True

    for key in (
        "type",
        "category",
        "expected_retrieval",
        "classification",
    ):
        value = normalize(
            case.get(key, "")
        )

        if value in {
            "unknown",
            "unanswerable",
            "negative",
            "out_of_scope",
        }:
            return True

    identifier = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    return identifier.startswith("unknown-")


def expected_targets(
    case: dict[str, Any],
) -> list[tuple[str, str]]:

    expected = case.get("expected")

    if isinstance(expected, list):

        targets = []

        for item in expected:

            if not isinstance(item, dict):
                continue

            source = normalize(
                item.get("source", "")
            )

            section = normalize(
                item.get("section", "")
            )

            if source or section:
                targets.append(
                    (source, section)
                )

        return targets

    source = normalize(
        case.get("expected_source")
        or case.get("source")
        or ""
    )

    section = normalize(
        case.get("expected_section")
        or case.get("section")
        or ""
    )

    if source or section:
        return [(source, section)]

    return []


def expected_sources(
    case: dict[str, Any],
) -> set[str]:

    return {
        source
        for source, _ in expected_targets(case)
        if source
    }


def expected_sections(
    case: dict[str, Any],
) -> set[str]:

    return {
        section
        for _, section in expected_targets(case)
        if section
    }


def is_multi_section_case(
    case: dict[str, Any],
) -> bool:

    targets = expected_targets(case)

    if len(targets) > 1:
        return True

    if normalize(
        case.get("type", "")
    ) == "multi":
        return True

    identifier = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    return identifier.startswith("multi-")


# -------------------------------------------------------------------
# Dense retrieval
# -------------------------------------------------------------------

def dense_retrieve(
    question: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:

    vector = embed_query(question)

    response = pinecone_index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=NAMESPACE,
    )

    results = []

    for match in response.get(
        "matches",
        [],
    ):

        metadata = (
            match.get("metadata")
            or {}
        )

        results.append(
            {
                "id": str(match["id"]),
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "section": metadata.get(
                    "section",
                    "unknown",
                ),
                "chunk_index": metadata.get(
                    "chunk_index",
                    -1,
                ),
                "score": float(
                    match.get(
                        "score",
                        0.0,
                    )
                ),
                "text": metadata.get(
                    "text",
                    "",
                ),
            }
        )

    return results


# -------------------------------------------------------------------
# Reranking
# -------------------------------------------------------------------

def rerank(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    global _last_rerank_time

    if not candidates:
        return []

    elapsed = (
        time.monotonic()
        - _last_rerank_time
    )

    if elapsed < RERANK_MIN_INTERVAL:
        time.sleep(
            RERANK_MIN_INTERVAL
            - elapsed
        )

    documents = [
        {
            "id": str(index),
            "text": candidate["text"],
        }
        for index, candidate
        in enumerate(candidates)
    ]

    response = pinecone_client.inference.rerank(
        model=settings.rerank_model,
        query=question,
        documents=documents,
        top_n=min(
            settings.rerank_top_n,
            len(documents),
        ),
        return_documents=False,
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

    return results


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def matches_target(
    result: dict[str, Any],
    targets: list[tuple[str, str]],
) -> bool:

    source = normalize(
        result.get("source", "")
    )

    section = normalize(
        result.get("section", "")
    )

    for target_source, target_section in targets:

        source_match = (
            not target_source
            or source == target_source
        )

        section_match = (
            not target_section
            or section == target_section
        )

        if source_match and section_match:
            return True

    return False


def evaluate_case(
    case: dict[str, Any],
    mode: str,
    index: int,
) -> dict[str, Any]:

    question = get_question(case)

    dense_results = dense_retrieve(
        question,
        top_k=5,
    )

    if mode == "dense":

        final_results = dense_results[:3]

    elif mode == "threshold":

        final_results = [
            result
            for result in dense_results
            if result["score"]
            >= settings.similarity_threshold
        ][:3]

    elif mode == "rerank":

        filtered = [
            result
            for result in dense_results
            if result["score"]
            >= settings.similarity_threshold
        ]

        final_results = rerank(
            question,
            filtered,
        )[:3]

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    unknown = is_unknown_case(case)
    multi = is_multi_section_case(case)
    targets = expected_targets(case)

    # ---------------------------------------------------------------
    # Unknown questions
    # ---------------------------------------------------------------

    if unknown:

        rejected = (
            len(final_results) == 0
        )

        return {
            "id": case_id(case, index),
            "question": question,
            "unknown": True,
            "multi_section": False,
            "top1_correct": rejected,
            "recall_at_3": rejected,
            "multi_section_recall": None,
            "results": final_results,
        }

    # ---------------------------------------------------------------
    # Known questions
    # ---------------------------------------------------------------

    if not final_results:

        return {
            "id": case_id(case, index),
            "question": question,
            "unknown": False,
            "multi_section": multi,
            "top1_correct": False,
            "recall_at_3": False,
            "multi_section_recall": (
                0.0
                if multi
                else None
            ),
            "results": [],
        }

    # Top-1
    top1_correct = matches_target(
        final_results[0],
        targets,
    )

    # Recall@3
    top3_results = final_results[:3]

    if multi and targets:

        retrieved_pairs = {
            (
                normalize(
                    result.get(
                        "source",
                        "",
                    )
                ),
                normalize(
                    result.get(
                        "section",
                        "",
                    )
                ),
            )
            for result in top3_results
        }

        recall_at_3 = all(
            target in retrieved_pairs
            for target in targets
        )

    else:

        recall_at_3 = any(
            matches_target(
                result,
                targets,
            )
            for result in top3_results
        )

    # Multi-section recall
    if multi and targets:

        retrieved_pairs = {
            (
                normalize(
                    result.get(
                        "source",
                        "",
                    )
                ),
                normalize(
                    result.get(
                        "section",
                        "",
                    )
                ),
            )
            for result in final_results[:5]
        }

        matched_targets = sum(
            target in retrieved_pairs
            for target in targets
        )

        multi_recall = (
            matched_targets
            / len(targets)
        )

    else:

        multi_recall = None

    return {
        "id": case_id(
            case,
            index,
        ),
        "question": question,
        "unknown": False,
        "multi_section": multi,
        "top1_correct": top1_correct,
        "recall_at_3": recall_at_3,
        "multi_section_recall": multi_recall,
        "results": final_results,
    }


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def calculate_metrics(
    results: list[dict[str, Any]],
) -> dict[str, float]:

    known = [
        result
        for result in results
        if not result["unknown"]
    ]

    unknown = [
        result
        for result in results
        if result["unknown"]
    ]

    multi = [
        result
        for result in results
        if result["multi_section"]
        and result["multi_section_recall"]
        is not None
    ]

    known_top1 = (
        sum(
            result["top1_correct"]
            for result in known
        )
        / len(known)
        if known
        else 0.0
    )

    known_recall = (
        sum(
            result["recall_at_3"]
            for result in known
        )
        / len(known)
        if known
        else 0.0
    )

    unknown_rejection = (
        sum(
            result["recall_at_3"]
            for result in unknown
        )
        / len(unknown)
        if unknown
        else 0.0
    )

    known_rejection = (
        sum(
            not result["results"]
            for result in known
        )
        / len(known)
        if known
        else 0.0
    )

    overall = (
        (
            sum(
                result["top1_correct"]
                for result in known
            )
            +
            sum(
                result["recall_at_3"]
                for result in unknown
            )
        )
        / len(results)
        if results
        else 0.0
    )

    multi_recall = (
        sum(
            result["multi_section_recall"]
            for result in multi
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
# Runner
# -------------------------------------------------------------------

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
        return

    print()
    print(
        f"--- {mode.upper()} FAILURES ---"
    )

    for result in failures:

        print()
        print(
            f"[{result['id']}] "
            f"{result['question']}"
        )

        print(
            f"Expected unknown: "
            f"{result['unknown']}"
        )

        print(
            f"Multi-section: "
            f"{result['multi_section']}"
        )

        if not result["results"]:

            print("Retrieved: NONE")
            continue

        for rank, retrieved in enumerate(
            result["results"],
            start=1,
        ):

            details = [
                f"dense={retrieved['score']:.4f}"
            ]

            if "rerank_score" in retrieved:
                details.append(
                    f"rerank="
                    f"{retrieved['rerank_score']:.4f}"
                )

            print(
                f"  {rank}. "
                f"{retrieved['source']} | "
                f"{retrieved['section']} | "
                f"{' | '.join(details)}"
            )


def main() -> None:

    cases = load_cases()

    known_count = sum(
        not is_unknown_case(case)
        for case in cases
    )

    unknown_count = sum(
        is_unknown_case(case)
        for case in cases
    )

    multi_count = sum(
        is_multi_section_case(case)
        for case in cases
    )

    print()
    print("=" * 80)
    print(
        "ENTERPRISE RAG STRESS TEST"
    )
    print("=" * 80)

    print(
        f"Cases:               {len(cases)}"
    )

    print(
        f"Namespace:           {NAMESPACE}"
    )

    print(
        f"Known cases:         {known_count}"
    )

    print(
        f"Unknown cases:       {unknown_count}"
    )

    print(
        f"Multi-section cases: {multi_count}"
    )

    print(
        f"Threshold:           "
        f"{settings.similarity_threshold}"
    )

    print(
        f"Reranker:            "
        f"{settings.rerank_model}"
    )

    modes = [
    (
        "dense",
        "Dense Retrieval",
    ),
    (
        "threshold",
        "Dense + Threshold",
    ),
]

    if not SKIP_RERANK:
        modes.append(
        (
            "rerank",
            "Dense + Threshold + Reranking",
        )
    )

    metrics_table = {}

    for mode, display_name in modes:

        print()
        print("=" * 80)
        print(display_name)
        print("=" * 80)

        results = []

        for index, case in enumerate(cases):

            result = evaluate_case(
                case,
                mode,
                index,
            )

            results.append(result)

        metric = calculate_metrics(
            results
        )

        metrics_table[
            display_name
        ] = metric

        print()
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

        print_failures(
            results,
            display_name,
        )

    print()
    print("=" * 105)
    print("STRESS TEST SUMMARY")
    print("=" * 105)

    print(
        f"{'Configuration':<38}"
        f"{'Top-1':>12}"
        f"{'Recall@3':>14}"
        f"{'Unknown':>14}"
        f"{'Known Reject':>16}"
        f"{'Overall':>14}"
        f"{'Multi-Sec':>14}"
    )

    print("-" * 105)

    for name, metric in metrics_table.items():

        print(
            f"{name:<38}"
            f"{metric['known_top1'] * 100:>11.2f}%"
            f"{metric['known_recall_at_3'] * 100:>13.2f}%"
            f"{metric['unknown_rejection'] * 100:>13.2f}%"
            f"{metric['known_rejection'] * 100:>15.2f}%"
            f"{metric['overall_accuracy'] * 100:>13.2f}%"
            f"{metric['multi_section_recall'] * 100:>13.2f}%"
        )

    print("=" * 105)


if __name__ == "__main__":
    main()