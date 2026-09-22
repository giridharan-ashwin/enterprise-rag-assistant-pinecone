import json
import time
from pathlib import Path

from app.services.generation import generate_answer


ROOT = Path(__file__).resolve().parents[1]

CASES_FILE = ROOT / "tests" / "answer_eval_cases.json"
CACHE_FILE = ROOT / "results" / "retrieval_cache.json"
OUTPUT_FILE = ROOT / "results" / "answer_eval_results.json"

THRESHOLD = 0.50

# Same RRF configuration used by the hybrid retrieval experiments.
RRF_K = 60


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(value):
    return str(value or "").strip().lower()


def target_key(source, section):
    return (
        normalize(source),
        normalize(section),
    )


def context_target(context):
    return target_key(
        context.get("source", ""),
        context.get("section", ""),
    )


def expected_keys(case):
    return {
        target_key(
            item.get("source", ""),
            item.get("section", ""),
        )
        for item in case.get("expected", [])
    }


def build_question_cache(cache):
    """
    The retrieval cache is keyed by retrieval-case ID, while
    answer_eval_cases.json uses its own IDs.

    Match them using the question text.
    """

    result = {}

    for case_id, value in cache.items():
        question = normalize(value.get("question", ""))

        if question:
            result[question] = value

    return result


def dense_key(item):
    return item.get("id", "")


def sparse_key(item):
    return item.get("id", "")


def build_hybrid_results(dense_results, sparse_results):
    """
    Reciprocal Rank Fusion.

    RRF score = 1 / (RRF_K + rank)

    Rank starts at 1.
    """

    combined = {}

    for rank, item in enumerate(dense_results, start=1):
        item_id = dense_key(item)

        if not item_id:
            continue

        entry = combined.setdefault(
            item_id,
            {
                "id": item_id,
                "source": item.get("source", ""),
                "section": item.get("section", ""),
                "chunk_index": item.get("chunk_index", 0),
                "text": item.get("text", ""),
                "dense_score": item.get("score"),
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )

        entry["rrf_score"] += 1.0 / (RRF_K + rank)

    for rank, item in enumerate(sparse_results, start=1):
        item_id = sparse_key(item)

        if not item_id:
            continue

        entry = combined.setdefault(
            item_id,
            {
                "id": item_id,
                "source": item.get("source", ""),
                "section": item.get("section", ""),
                "chunk_index": item.get("chunk_index", 0),
                "text": item.get("text", ""),
                "dense_score": None,
                "bm25_score": item.get("bm25_score"),
                "rrf_score": 0.0,
            },
        )

        entry["bm25_score"] = item.get("bm25_score")
        entry["rrf_score"] += 1.0 / (RRF_K + rank)

    results = list(combined.values())

    results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return results


def apply_dense_threshold(results):
    """
    Threshold is applied to the original dense similarity score.

    This matches the research experiment where the threshold
    represents semantic similarity rather than the RRF score.
    """

    filtered = []

    for item in results:
        dense_score = item.get("dense_score")

        if dense_score is None:
            continue

        if float(dense_score) >= THRESHOLD:
            filtered.append(item)

    return filtered


def prepare_contexts(hybrid_results):
    """
    Convert hybrid retrieval results into the structure expected
    by the generation service.
    """

    contexts = []

    for item in hybrid_results:
        contexts.append(
            {
                "id": item.get("id", ""),
                "source": item.get("source", ""),
                "section": item.get("section", ""),
                "chunk_index": item.get("chunk_index", 0),
                "score": item.get("dense_score", 0.0),
                "rrf_score": item.get("rrf_score", 0.0),
                "text": item.get("text", ""),
            }
        )

    return contexts


def evaluate_retrieval(case, contexts):
    expected = expected_keys(case)

    retrieved = {
        context_target(context)
        for context in contexts
    }

    if not expected:
        return {
            "retrieval_support": (
                1.0 if not contexts else 0.0
            ),
            "retrieved_expected": [],
        }

    matched = expected.intersection(retrieved)

    return {
        "retrieval_support": len(matched) / len(expected),
        "retrieved_expected": [
            {
                "source": source,
                "section": section,
            }
            for source, section in sorted(matched)
        ],
    }


def citation_keys(result):
    citations = result.get("citations", [])

    return {
        target_key(
            citation.get("source", ""),
            citation.get("section", ""),
        )
        for citation in citations
    }


def evaluate_citations(case, result):
    expected = expected_keys(case)
    citations = citation_keys(result)

    if case["type"] == "unknown":
        return {
            "citation_accuracy": (
                1.0 if not citations else 0.0
            )
        }

    if not citations:
        return {
            "citation_accuracy": 0.0
        }

    matched = citations.intersection(expected)

    return {
        "citation_accuracy": (
            len(matched) / len(expected)
            if expected
            else 0.0
        )
    }


def evaluate_abstention(case, result):
    if case["type"] != "unknown":
        return {
            "abstention_correct": None
        }

    return {
        "abstention_correct": bool(
            result.get("abstained", False)
        )
    }


def main():
    cases = load_json(CASES_FILE)
    cache = load_json(CACHE_FILE)

    question_cache = build_question_cache(cache)

    print(f"Answer benchmark cases: {len(cases)}")
    print(f"Retrieval cache entries: {len(cache)}")
    print(f"Threshold: {THRESHOLD}")
    print(f"RRF K: {RRF_K}")
    print()

    results = []

    missing_cases = []

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        question = case["question"]

        print(
            f"[{index}/{len(cases)}] "
            f"{case_id}: {question}"
        )

        cached = question_cache.get(
            normalize(question)
        )

        if cached is None:
            print("  WARNING: no matching cache entry")
            missing_cases.append(case_id)
            continue

        dense_results = cached.get("dense", [])
        sparse_results = cached.get("sparse", [])

        hybrid_results = build_hybrid_results(
            dense_results,
            sparse_results,
        )

        threshold_results = apply_dense_threshold(
            hybrid_results
        )

        contexts = prepare_contexts(
            threshold_results
        )

        retrieval_metrics = evaluate_retrieval(
            case,
            contexts,
        )

        start = time.perf_counter()

        answer_result = generate_answer(
            question,
            contexts,
        )

        generation_ms = (
            time.perf_counter() - start
        ) * 1000

        citation_metrics = evaluate_citations(
            case,
            answer_result,
        )

        abstention_metrics = evaluate_abstention(
            case,
            answer_result,
        )

        result = {
            "id": case_id,
            "type": case["type"],
            "question": question,

            "retrieval": {
                "strategy": "hybrid_rrf",
                "threshold": THRESHOLD,
                "dense_candidate_count": len(
                    dense_results
                ),
                "sparse_candidate_count": len(
                    sparse_results
                ),
                "hybrid_candidate_count": len(
                    hybrid_results
                ),
                "threshold_context_count": len(
                    contexts
                ),
            },

            "retrieved_contexts": contexts,

            "answer": answer_result,

            "retrieval_metrics": retrieval_metrics,

            "citation_metrics": citation_metrics,

            "abstention_metrics": abstention_metrics,

            "generation_latency_ms": round(
                generation_ms,
                2,
            ),
        }

        results.append(result)

        print(
            f"  contexts={len(contexts)} "
            f"retrieval_support="
            f"{retrieval_metrics['retrieval_support']:.2f} "
            f"citations="
            f"{len(answer_result.get('citations', []))} "
            f"abstained="
            f"{answer_result.get('abstained', False)}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("========================================")
    print("Answer Evaluation Complete")
    print("========================================")
    print(f"Completed: {len(results)}")
    print(f"Missing:   {len(missing_cases)}")

    if missing_cases:
        print()
        print("Missing cases:")
        for case_id in missing_cases:
            print(f"  - {case_id}")

    print()
    print(f"Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()