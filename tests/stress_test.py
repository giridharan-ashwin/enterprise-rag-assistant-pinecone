from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Any

from app.clients import pinecone_client, pinecone_index
from app.config import settings
from app.services.embeddings import embed_query


STRESS_NAMESPACE = "stress-test"
CASES_FILE = Path(__file__).with_name("stress_test_cases.json")
RERANK_MIN_INTERVAL = 1.1
_last_rerank_time = 0.0


def load_cases() -> list[dict[str, Any]]:
    """Load evaluation cases from tests/stress_test_cases.json."""
    with CASES_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("cases", "questions", "test_cases", "queries"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        f"Unsupported test case format in {CASES_FILE}. "
        "Expected a list or a dictionary containing a list."
    )

def rate_limit_rerank():
    global _last_rerank_time

    now = time.monotonic()
    elapsed = now - _last_rerank_time

    if elapsed < RERANK_MIN_INTERVAL:
        time.sleep(RERANK_MIN_INTERVAL - elapsed)

    _last_rerank_time = time.monotonic()

def normalize(value: Any) -> str:
    return str(value).strip().lower()


def case_id(case: dict[str, Any], index: int) -> str:
    return str(
        case.get("id")
        or case.get("question_id")
        or f"Q{index + 1}"
    )


def get_question(case: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt"):
        value = case.get(key)
        if value:
            return str(value).strip()

    raise ValueError(f"No question field found in case: {case}")


def is_unknown_case(case: dict[str, Any]) -> bool:
    """
    Supports several common labels:
      known=false
      type=unknown
      category=unknown
      expected_retrieval=unknown
    """
    if case.get("known") is False:
        return True

    for key in ("type", "category", "expected_retrieval"):
        value = normalize(case.get(key, ""))
        if value in {"unknown", "unanswerable", "negative", "out_of_scope"}:
            return True

    return False


def expected_sections(case: dict[str, Any]) -> set[str]:
    """
    Supports:
      expected_sections: ["Remote Work", "Security"]
      expected_section: "Remote Work"
      section: "Remote Work"
    """
    value = (
        case.get("expected_sections")
        or case.get("expected_section")
        or case.get("section")
        or []
    )

    if isinstance(value, str):
        return {normalize(value)}

    if isinstance(value, list):
        return {normalize(item) for item in value if item}

    return set()


def expected_sources(case: dict[str, Any]) -> set[str]:
    """
    Supports:
      expected_sources: ["hr_remote_work.md"]
      expected_source: "hr_remote_work.md"
      source: "hr_remote_work.md"
    """
    value = (
        case.get("expected_sources")
        or case.get("expected_source")
        or case.get("source")
        or []
    )

    if isinstance(value, str):
        return {normalize(value)}

    if isinstance(value, list):
        return {normalize(item) for item in value if item}

    return set()


def query_dense(question: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Pure dense retrieval. No threshold and no reranking."""
    vector = embed_query(question)

    response = pinecone_index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        namespace=STRESS_NAMESPACE,
    )

    results: list[dict[str, Any]] = []

    for match in response.get("matches", []):
        metadata = match.get("metadata", {}) or {}

        results.append(
            {
                "source": metadata.get("source", "unknown"),
                "section": metadata.get("section", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": float(match.get("score", 0.0)),
                "text": metadata.get("text", ""),
            }
        )

    return results


def apply_threshold(
    results: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Filter dense results using the configured similarity threshold."""
    return [
        result
        for result in results
        if float(result["score"]) >= threshold
    ]


def rerank(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rerank candidates using Pinecone's hosted reranker."""
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

    results: list[dict[str, Any]] = []

    for item in response.data:
        candidate = candidates[item.index]

        results.append(
            {
                **candidate,
                "rerank_score": float(item.score),
            }
        )

    return results


def evaluate_case(
    case: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    question = get_question(case)

    # Retrieve more candidates for the baseline so thresholding
    # and reranking have enough material to work with.
    dense_results = query_dense(question, top_k=5)

    if mode == "dense":
        final_results = dense_results[:3]

    elif mode == "threshold":
        filtered = apply_threshold(
            dense_results,
            settings.similarity_threshold,
        )
        final_results = filtered[:3]

    elif mode == "rerank":
        filtered = apply_threshold(
            dense_results,
            settings.similarity_threshold,
        )
        final_results = rerank(question, filtered)
        final_results = final_results[:3]

    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

    expected_section_set = expected_sections(case)
    expected_source_set = expected_sources(case)
    unknown = is_unknown_case(case)

    returned_sections = {
        normalize(result["section"])
        for result in final_results
    }

    returned_sources = {
        normalize(result["source"])
        for result in final_results
    }

    # Top-1:
    # For known queries, top result must match the expected source or section.
    # For unknown queries, top-1 is considered rejected when no result survives.
    if unknown:
        top1_correct = len(final_results) == 0
    else:
        if not final_results:
            top1_correct = False
        else:
            top = final_results[0]

            top_section = normalize(top["section"])
            top_source = normalize(top["source"])

            section_match = (
                not expected_section_set
                or top_section in expected_section_set
            )

            source_match = (
                not expected_source_set
                or top_source in expected_source_set
            )

            top1_correct = section_match and source_match

    # Recall@3:
    # Known query is successful when one of the returned results matches
    # the expected section/source. Unknown query is successful when rejected.
    if unknown:
        recall_at_3 = len(final_results) == 0
    else:
        section_match = (
            not expected_section_set
            or bool(returned_sections & expected_section_set)
        )

        source_match = (
            not expected_source_set
            or bool(returned_sources & expected_source_set)
        )

        recall_at_3 = section_match and source_match

    # Multi-section recall:
    # Only applies when the evaluation case expects >1 section.
    if len(expected_section_set) > 1:
        multi_section_hits = len(
            returned_sections & expected_section_set
        )
        multi_section_recall = (
            multi_section_hits / len(expected_section_set)
        )
    else:
        multi_section_recall = None

    return {
        "id": case_id(case, 0),
        "question": question,
        "unknown": unknown,
        "results": final_results,
        "top1_correct": top1_correct,
        "recall_at_3": recall_at_3,
        "multi_section_recall": multi_section_recall,
    }


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {
            "top1_accuracy": 0.0,
            "recall_at_3": 0.0,
            "unknown_rejection": 0.0,
            "multi_section_recall": 0.0,
        }

    top1_accuracy = sum(
        result["top1_correct"] for result in results
    ) / len(results)

    recall_at_3 = sum(
        result["recall_at_3"] for result in results
    ) / len(results)

    unknown_results = [
        result for result in results
        if result["unknown"]
    ]

    if unknown_results:
        unknown_rejection = sum(
            result["recall_at_3"]
            for result in unknown_results
        ) / len(unknown_results)
    else:
        unknown_rejection = 0.0

    multi_section_results = [
        result["multi_section_recall"]
        for result in results
        if result["multi_section_recall"] is not None
    ]

    if multi_section_results:
        multi_section_recall = (
            sum(multi_section_results)
            / len(multi_section_results)
        )
    else:
        multi_section_recall = 0.0

    return {
        "top1_accuracy": top1_accuracy,
        "recall_at_3": recall_at_3,
        "unknown_rejection": unknown_rejection,
        "multi_section_recall": multi_section_recall,
    }


def print_metrics(
    mode: str,
    metrics: dict[str, float],
) -> None:
    print()
    print("=" * 70)
    print(mode.upper())
    print("=" * 70)

    print(
        f"Top-1 Accuracy:       "
        f"{metrics['top1_accuracy'] * 100:.2f}%"
    )
    print(
        f"Recall@3:             "
        f"{metrics['recall_at_3'] * 100:.2f}%"
    )
    print(
        f"Unknown Rejection:    "
        f"{metrics['unknown_rejection'] * 100:.2f}%"
    )
    print(
        f"Multi-Section Recall: "
        f"{metrics['multi_section_recall'] * 100:.2f}%"
    )


def print_failures(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
    mode: str,
) -> None:
    failures = [
        (case, result)
        for case, result in zip(cases, results)
        if not result["top1_correct"]
    ]

    if not failures:
        return

    print()
    print(f"--- {mode.upper()} FAILURES ---")

    for case, result in failures:
        print()
        print(f"[{result['id']}] {result['question']}")
        print(f"Expected unknown: {result['unknown']}")

        expected_sections_value = (
            case.get("expected_sections")
            or case.get("expected_section")
            or case.get("section")
        )

        expected_sources_value = (
            case.get("expected_sources")
            or case.get("expected_source")
            or case.get("source")
        )

        if expected_sections_value:
            print(f"Expected section(s): {expected_sections_value}")

        if expected_sources_value:
            print(f"Expected source(s):  {expected_sources_value}")

        if not result["results"]:
            print("Retrieved: NONE")
            continue

        for rank, retrieved in enumerate(result["results"], start=1):
            score = retrieved["score"]

            if "rerank_score" in retrieved:
                print(
                    f"  {rank}. "
                    f"{retrieved['source']} | "
                    f"{retrieved['section']} | "
                    f"dense={score:.4f} | "
                    f"rerank={retrieved['rerank_score']:.4f}"
                )
            else:
                print(
                    f"  {rank}. "
                    f"{retrieved['source']} | "
                    f"{retrieved['section']} | "
                    f"score={score:.4f}"
                )


def print_summary_table(
    all_metrics: dict[str, dict[str, float]],
) -> None:
    print()
    print("=" * 92)
    print("STRESS TEST SUMMARY")
    print("=" * 92)

    header = (
        f"{'Configuration':<32}"
        f"{'Top-1':>12}"
        f"{'Recall@3':>12}"
        f"{'Unknown':>12}"
        f"{'Multi-Sec':>12}"
    )

    print(header)
    print("-" * 92)

    for name, metrics in all_metrics.items():
        print(
            f"{name:<32}"
            f"{metrics['top1_accuracy'] * 100:>11.2f}%"
            f"{metrics['recall_at_3'] * 100:>11.2f}%"
            f"{metrics['unknown_rejection'] * 100:>11.2f}%"
            f"{metrics['multi_section_recall'] * 100:>11.2f}%"
        )

    print("=" * 92)


def main() -> None:
    cases = load_cases()

    print()
    print("=" * 70)
    print("ENTERPRISE RAG BIG-BANG STRESS TEST")
    print("=" * 70)
    print(f"Cases:                {len(cases)}")
    print(f"Namespace:             {STRESS_NAMESPACE}")
    print(f"Similarity threshold:  {settings.similarity_threshold}")
    print(f"Reranker:              {settings.rerank_model}")
    print(f"Rerank top N:          {settings.rerank_top_n}")

    known_count = sum(
        1 for case in cases
        if not is_unknown_case(case)
    )

    unknown_count = len(cases) - known_count

    multi_section_count = sum(
        1 for case in cases
        if len(expected_sections(case)) > 1
    )

    print(f"Known cases:           {known_count}")
    print(f"Unknown cases:         {unknown_count}")
    print(f"Multi-section cases:   {multi_section_count}")

    modes = [
        ("dense", "Dense Retrieval"),
        ("threshold", "Dense + Threshold"),
        ("rerank", "Dense + Threshold + Reranking"),
    ]

    all_metrics: dict[str, dict[str, float]] = {}

    for mode, display_name in modes:
        print()
        print(f"Running: {display_name}")

        results: list[dict[str, Any]] = []

        for index, case in enumerate(cases):
            result = evaluate_case(case, mode)

            # Fix ID for datasets without explicit IDs.
            if not case.get("id") and not case.get("question_id"):
                result["id"] = f"Q{index + 1}"

            results.append(result)

        metrics = calculate_metrics(results)
        all_metrics[display_name] = metrics

        print_metrics(display_name, metrics)
        print_failures(cases, results, display_name)

    print_summary_table(all_metrics)


if __name__ == "__main__":
    main()