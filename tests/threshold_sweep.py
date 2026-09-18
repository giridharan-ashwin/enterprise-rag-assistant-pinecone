from __future__ import annotations

from app.config import settings
from tests.stress_test import (
    expected_sections,
    expected_sources,
    get_question,
    is_unknown_case,
    load_cases,
    query_dense,
)


THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


def evaluate_threshold(cases, threshold):
    known_cases = [
        case for case in cases
        if not is_unknown_case(case)
    ]

    unknown_cases = [
        case for case in cases
        if is_unknown_case(case)
    ]

    cached = []

    for case in cases:
        question = get_question(case)

        dense_results = query_dense(
            question,
            top_k=5,
        )

        filtered = [
            result
            for result in dense_results
            if float(result["score"]) >= threshold
        ][:3]

        cached.append(
            {
                "case": case,
                "results": filtered,
            }
        )

    known_top1_correct = 0
    known_recall_at_3_correct = 0
    known_rejected = 0

    unknown_rejected = 0
    unknown_accepted = 0

    for item in cached:
        case = item["case"]
        results = item["results"]

        if is_unknown_case(case):
            if not results:
                unknown_rejected += 1
            else:
                unknown_accepted += 1

            continue

        if not results:
            known_rejected += 1
            continue

        expected_source_set = expected_sources(case)
        expected_section_set = expected_sections(case)

        returned_sources = {
            str(result["source"]).strip().lower()
            for result in results
        }

        returned_sections = {
            str(result["section"]).strip().lower()
            for result in results
        }

        top = results[0]

        top_source = str(top["source"]).strip().lower()
        top_section = str(top["section"]).strip().lower()

        source_top_match = (
            not expected_source_set
            or top_source in expected_source_set
        )

        section_top_match = (
            not expected_section_set
            or top_section in expected_section_set
        )

        if source_top_match and section_top_match:
            known_top1_correct += 1

        source_recall_match = (
            not expected_source_set
            or bool(returned_sources & expected_source_set)
        )

        section_recall_match = (
            not expected_section_set
            or bool(returned_sections & expected_section_set)
        )

        if source_recall_match and section_recall_match:
            known_recall_at_3_correct += 1

    known_count = len(known_cases)
    unknown_count = len(unknown_cases)
    total = len(cases)

    known_top1 = (
        known_top1_correct / known_count
        if known_count
        else 0.0
    )

    known_recall_at_3 = (
        known_recall_at_3_correct / known_count
        if known_count
        else 0.0
    )

    unknown_rejection = (
        unknown_rejected / unknown_count
        if unknown_count
        else 0.0
    )

    known_rejection = (
        known_rejected / known_count
        if known_count
        else 0.0
    )

    overall_decision_accuracy = (
        (known_top1_correct + unknown_rejected) / total
        if total
        else 0.0
    )

    return {
        "threshold": threshold,
        "known_top1": known_top1,
        "known_recall_at_3": known_recall_at_3,
        "unknown_rejection": unknown_rejection,
        "known_rejection": known_rejection,
        "overall_decision_accuracy": overall_decision_accuracy,
    }


def main():
    cases = load_cases()

    known_count = sum(
        not is_unknown_case(case)
        for case in cases
    )

    unknown_count = sum(
        is_unknown_case(case)
        for case in cases
    )

    print()
    print("=" * 110)
    print("RAG SIMILARITY THRESHOLD SWEEP")
    print("=" * 110)
    print(f"Cases:             {len(cases)}")
    print(f"Known cases:        {known_count}")
    print(f"Unknown cases:      {unknown_count}")
    print(f"Current threshold:  {settings.similarity_threshold}")
    print("=" * 110)

    results = []

    for threshold in THRESHOLDS:
        print(
            f"Evaluating threshold {threshold:.2f} ..."
        )

        result = evaluate_threshold(
            cases,
            threshold,
        )

        results.append(result)

    print()
    print("=" * 125)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 125)

    print(
        f"{'Threshold':>10}"
        f"{'Known Top-1':>15}"
        f"{'Known Recall@3':>18}"
        f"{'Unknown Reject':>18}"
        f"{'Known Reject':>16}"
        f"{'Overall Acc':>16}"
    )

    print("-" * 125)

    for result in results:
        print(
            f"{result['threshold']:>10.2f}"
            f"{result['known_top1'] * 100:>14.2f}%"
            f"{result['known_recall_at_3'] * 100:>17.2f}%"
            f"{result['unknown_rejection'] * 100:>17.2f}%"
            f"{result['known_rejection'] * 100:>15.2f}%"
            f"{result['overall_decision_accuracy'] * 100:>15.2f}%"
        )

    print("=" * 125)

    print()
    print(
        "Interpretation:"
    )
    print(
        "  Known Top-1 / Recall@3 measure retrieval quality "
        "for answerable questions."
    )
    print(
        "  Unknown Rejection measures abstention quality."
    )
    print(
        "  Known Rejection measures how often thresholding "
        "incorrectly discards answerable questions."
    )
    print(
        "  Overall Accuracy counts both successful answers "
        "and correctly rejected unknown questions."
    )


if __name__ == "__main__":
    main()