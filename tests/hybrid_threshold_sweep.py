from __future__ import annotations

from tests.eval_utils import (
    get_question,
    is_unknown_case,
    retrieval_matches_case,
)
from tests.hybrid_eval import dense_retrieve
from tests.stress_test import load_cases
from app.services.hybrid import (
    bm25_retriever,
    reciprocal_rank_fusion,
)


THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
]


DENSE_TOP_K = 10
SPARSE_TOP_K = 10
FINAL_TOP_K = 5


def build_cache(cases):
    cache = []

    for index, case in enumerate(cases):
        question = get_question(case)

        dense = dense_retrieve(
            question,
            top_k=DENSE_TOP_K,
        )

        sparse = bm25_retriever.retrieve(
            question,
            top_k=SPARSE_TOP_K,
        )

        cache.append(
            {
                "index": index,
                "case": case,
                "dense": dense,
                "sparse": sparse,
            }
        )

    return cache


def evaluate(cache, threshold):
    known = [
        item for item in cache
        if not is_unknown_case(item["case"])
    ]

    unknown = [
        item for item in cache
        if is_unknown_case(item["case"])
    ]

    known_top1 = 0
    known_recall = 0
    known_rejected = 0

    unknown_rejected = 0

    for item in cache:
        case = item["case"]

        dense = [
            result
            for result in item["dense"]
            if float(result["score"]) >= threshold
        ]

        # For the threshold experiment we require dense evidence
        # before allowing BM25 to contribute.
        if not dense:
            results = []
        else:
            results = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=item["sparse"],
                top_k=FINAL_TOP_K,
            )

        if is_unknown_case(case):
            if not results:
                unknown_rejected += 1

            continue

        if not results:
            known_rejected += 1
            continue

        top1, recall = retrieval_matches_case(
            case,
            results,
        )

        if top1:
            known_top1 += 1

        if recall:
            known_recall += 1

    known_count = len(known)
    unknown_count = len(unknown)
    total = len(cache)

    unknown_acceptance = (
        unknown_count - unknown_rejected
    )

    overall_correct = (
        known_top1
        + unknown_rejected
    )

    return {
        "threshold": threshold,
        "known_top1": (
            known_top1 / known_count
            if known_count
            else 0.0
        ),
        "known_recall_at_3": (
            known_recall / known_count
            if known_count
            else 0.0
        ),
        "unknown_rejection": (
            unknown_rejected / unknown_count
            if unknown_count
            else 0.0
        ),
        "known_rejection": (
            known_rejected / known_count
            if known_count
            else 0.0
        ),
        "unknown_false_positive": (
            unknown_acceptance / unknown_count
            if unknown_count
            else 0.0
        ),
        "overall_accuracy": (
            overall_correct / total
            if total
            else 0.0
        ),
    }


def main():
    cases = load_cases()

    print()
    print("=" * 120)
    print("HYBRID RETRIEVAL THRESHOLD SWEEP")
    print("=" * 120)

    print(f"Cases:          {len(cases)}")
    print(
        f"Known:          "
        f"{sum(not is_unknown_case(c) for c in cases)}"
    )
    print(
        f"Unknown:        "
        f"{sum(is_unknown_case(c) for c in cases)}"
    )

    cache = build_cache(cases)

    results = []

    for threshold in THRESHOLDS:
        print(
            f"Evaluating hybrid threshold "
            f"{threshold:.2f} ..."
        )

        result = evaluate(
            cache,
            threshold,
        )

        results.append(result)

    print()
    print("=" * 120)
    print("RESULTS")
    print("=" * 120)

    print(
        f"{'Threshold':>10}"
        f"{'Known Top-1':>15}"
        f"{'Recall@3':>15}"
        f"{'Unknown Reject':>18}"
        f"{'Known Reject':>16}"
        f"{'Overall':>14}"
    )

    print("-" * 120)

    for result in results:
        print(
            f"{result['threshold']:>10.2f}"
            f"{result['known_top1'] * 100:>14.2f}%"
            f"{result['known_recall_at_3'] * 100:>14.2f}%"
            f"{result['unknown_rejection'] * 100:>17.2f}%"
            f"{result['known_rejection'] * 100:>15.2f}%"
            f"{result['overall_accuracy'] * 100:>13.2f}%"
        )

    print("=" * 120)

    best = max(
        results,
        key=lambda result: result["overall_accuracy"],
    )

    print()
    print(
        f"Highest observed overall accuracy: "
        f"{best['threshold']:.2f}"
    )

    print(
        "This is dataset-specific and should not "
        "be treated as a universal threshold."
    )


if __name__ == "__main__":
    main()
