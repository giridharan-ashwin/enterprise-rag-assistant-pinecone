from __future__ import annotations

import json
import time
from pathlib import Path

from app.clients import pinecone_client
from app.config import settings

from app.services.hybrid import (
    bm25_retriever,
    reciprocal_rank_fusion,
)

from tests.eval_utils import (
    get_question,
    is_unknown_case,
    multi_section_recall,
    retrieval_matches_case,
)

from tests.hybrid_eval import dense_retrieve
from tests.stress_test import load_cases


NAMESPACE = "stress-test"

THRESHOLD = 0.50

DENSE_TOP_K = 10
SPARSE_TOP_K = 10
FINAL_TOP_K = 5

RERANK_MIN_INTERVAL = 1.1
_last_rerank_time = 0.0

OUTPUT_FILE = Path(
    "results/research_benchmark.json"
)


def rate_limit_rerank():
    global _last_rerank_time

    now = time.monotonic()
    elapsed = now - _last_rerank_time

    if elapsed < RERANK_MIN_INTERVAL:
        time.sleep(
            RERANK_MIN_INTERVAL - elapsed
        )

    _last_rerank_time = time.monotonic()


def rerank(question, results):
    if not results:
        return []

    documents = [
        {
            "id": str(index),
            "text": result["text"],
        }
        for index, result
        in enumerate(results)
    ]

    rate_limit_rerank()

    response = 
pinecone_client.inference.rerank(
        model=settings.rerank_model,
        query=question,
        documents=documents,
        top_n=min(
            settings.rerank_top_n,
            len(documents),
        ),
        return_documents=False,
    )

    output = []

    for item in response.data:
        result = results[item.index]

        output.append(
            {
                **result,
                "rerank_score": float(
                    item.score
                ),
            }
        )

    return output


def hybrid(
    dense,
    sparse,
):
    return reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        top_k=FINAL_TOP_K,
    )


def hybrid_threshold(
    dense,
    sparse,
):
    dense = [
        result
        for result in dense
        if float(result["score"]) >= THRESHOLD
    ]

    if not dense:
        return []

    return hybrid(
        dense,
        sparse,
    )


def evaluate_case(
    case,
    mode,
    dense,
    sparse,
):
    question = get_question(case)

    if mode == "dense":
        results = dense[:FINAL_TOP_K]

    elif mode == "dense_threshold":
        results = [
            result
            for result in dense
            if float(result["score"]) >= 
THRESHOLD
        ][:FINAL_TOP_K]

    elif mode == "dense_threshold_rerank":
        candidates = [
            result
            for result in dense
            if float(result["score"]) >= 
THRESHOLD
        ]

        results = rerank(
            question,
            candidates,
        )

    elif mode == "hybrid":
        results = hybrid(
            dense,
            sparse,
        )

    elif mode == "hybrid_threshold":
        results = hybrid_threshold(
            dense,
            sparse,
        )

    elif mode == "hybrid_threshold_rerank":
        candidates = hybrid_threshold(
            dense,
            sparse,
        )

        results = rerank(
            question,
            candidates,
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    top1, recall = retrieval_matches_case(
        case,
        results,
    )

    multi_recall = multi_section_recall(
        case,
        results,
    )

    return {
        "top1": top1,
        "recall": recall,
        "multi_section_recall": multi_recall,
        "unknown": is_unknown_case(case),
        "results": results,
    }


def metrics(results):
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

    known_top1 = (
        sum(
            result["top1"]
            for result in known
        )
        / len(known)
        if known
        else 0.0
    )

    known_recall = (
        sum(
            result["recall"]
            for result in known
        )
        / len(known)
        if known
        else 0.0
    )

    unknown_rejection = (
        sum(
            result["recall"]
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
                result["top1"]
                for result in known
            )
            +
            sum(
                result["recall"]
                for result in unknown
            )
        )
        / len(results)
        if results
        else 0.0
    )

    multi_values = [
        result["multi_section_recall"]
        for result in results
        if result["multi_section_recall"]
        is not None
    ]

    multi_section = (
        sum(multi_values)
        / len(multi_values)
        if multi_values
        else 0.0
    )

    return {
        "known_top1": known_top1,
        "known_recall_at_3": known_recall,
        "unknown_rejection": unknown_rejection,
        "known_rejection": known_rejection,
        "overall_accuracy": overall,
        "multi_section_recall": multi_section,
    }


def main():
    cases = load_cases()

    print()
    print("=" * 120)
    print("ENTERPRISE RAG RESEARCH BENCHMARK")
    print("=" * 120)

    print(f"Cases:                
{len(cases)}")
    print(f"Namespace:            {NAMESPACE}")
    print(f"Threshold:            {THRESHOLD}")
    print(f"Reranker:             
{settings.rerank_model}")

    modes = [
        (
            "dense",
            "Dense",
        ),
        (
            "dense_threshold",
            "Dense + Threshold",
        ),
        (
            "dense_threshold_rerank",
            "Dense + Threshold + Rerank",
        ),
        (
            "hybrid",
            "Hybrid",
        ),
        (
            "hybrid_threshold",
            "Hybrid + Threshold",
        ),
        (
            "hybrid_threshold_rerank",
            "Hybrid + Threshold + Rerank",
        ),
    ]

    cache = []

    print()
    print("Building retrieval cache...")

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

    all_metrics = {}
    all_results = {}

    for mode, display_name in modes:

        print()
        print("=" * 120)
        print(f"RUNNING: {display_name}")
        print("=" * 120)

        mode_results = []

        for item in cache:

            result = evaluate_case(
                item["case"],
                mode,
                item["dense"],
                item["sparse"],
            )

            result["id"] = (
                item["case"].get("id")
                or f"Q{item['index'] + 1}"
            )

            result["question"] = get_question(
                item["case"]
            )

            mode_results.append(result)

        all_results[display_name] = 
mode_results

        metric = metrics(
            mode_results
        )

        all_metrics[display_name] = metric

        print(
            f"Known Top-1:          "
            f"{metric['known_top1'] * 
100:.2f}%"
        )

        print(
            f"Known Recall@3:      "
            f"{metric['known_recall_at_3'] * 
100:.2f}%"
        )

        print(
            f"Unknown Rejection:   "
            f"{metric['unknown_rejection'] * 
100:.2f}%"
        )

        print(
            f"Known Rejection:     "
            f"{metric['known_rejection'] * 
100:.2f}%"
        )

        print(
            f"Overall Accuracy:    "
            f"{metric['overall_accuracy'] * 
100:.2f}%"
        )

        print(
            f"Multi-Section Recall:"
            f" {metric['multi_section_recall'] 
* 100:.2f}%"
        )

    print()
    print("=" * 130)
    print("FINAL RESEARCH COMPARISON")
    print("=" * 130)

    print(
        f"{'Configuration':<38}"
        f"{'Known Top-1':>14}"
        f"{'Recall@3':>14}"
        f"{'Unknown':>14}"
        f"{'Known Reject':>16}"
        f"{'Overall':>14}"
        f"{'Multi-Sec':>14}"
    )

    print("-" * 130)

    for name, metric in all_metrics.items():

        print(
            f"{name:<38}"
            f"{metric['known_top1'] * 
100:>13.2f}%"
            f"{metric['known_recall_at_3'] * 
100:>13.2f}%"
            f"{metric['unknown_rejection'] * 
100:>13.2f}%"
            f"{metric['known_rejection'] * 
100:>15.2f}%"
            f"{metric['overall_accuracy'] * 
100:>13.2f}%"
            f"{metric['multi_section_recall'] * 
100:>13.2f}%"
        )

    print("=" * 130)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "namespace": NAMESPACE,
                "threshold": THRESHOLD,
                "reranker": 
settings.rerank_model,
                "metrics": all_metrics,
                "results": all_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Saved results to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
