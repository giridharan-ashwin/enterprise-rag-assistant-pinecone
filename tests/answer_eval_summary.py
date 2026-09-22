import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

JUDGE_FILE = ROOT / "results" / "answer_judge_results.json"
OUTPUT_FILE = ROOT / "results" / "answer_eval_summary.json"


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def percentage(value, total):
    if total == 0:
        return 0.0

    return round(
        (value / total) * 100,
        2,
    )


def average_score(results, field):
    values = [
        result["judge"].get(field, 0)
        for result in results
    ]

    if not values:
        return 0.0

    return round(mean(values), 3)


def score_percentage(results, field, maximum=2):
    if not results:
        return 0.0

    total = sum(
        result["judge"].get(field, 0)
        for result in results
    )

    maximum_total = len(results) * maximum

    return round(
        (total / maximum_total) * 100,
        2,
    )


def main():
    results = load_json(JUDGE_FILE)

    known = [
        result
        for result in results
        if result["type"] == "known"
    ]

    multi = [
        result
        for result in results
        if result["type"] == "multi"
    ]

    unknown = [
        result
        for result in results
        if result["type"] == "unknown"
    ]

    all_answerable = known + multi

    # Answer quality for answerable questions.
    correctness = score_percentage(
        all_answerable,
        "correctness",
    )

    faithfulness = score_percentage(
        all_answerable,
        "faithfulness",
    )

    citation_accuracy = score_percentage(
        all_answerable,
        "citation_accuracy",
    )

    # Multi-section performance.
    multi_correct = sum(
        result["judge"].get("correctness", 0) == 2
        for result in multi
    )

    multi_faithful = sum(
        result["judge"].get("faithfulness", 0) == 2
        for result in multi
    )

    multi_citation = sum(
        result["judge"].get("citation_accuracy", 0) == 2
        for result in multi
    )

    # Unknown-question behavior.
    #
    # The generated answer is considered correctly abstained when:
    # - the system says it abstained
    # - no citations are returned
    #
    # We intentionally do not use judge correctness here.
    unknown_abstained = sum(
        result["answer"].get("abstained", False)
        for result in unknown
    )

    unknown_with_citations = sum(
        bool(result["answer"].get("citations"))
        for result in unknown
    )

    unknown_abstention_rate = percentage(
        unknown_abstained,
        len(unknown),
    )

    unknown_citation_rate = percentage(
        unknown_with_citations,
        len(unknown),
    )

    # Generation latency.
    latencies = [
        result.get("judge_latency_ms", 0)
        for result in results
    ]

    generation_latencies = [
        result.get(
            "answer",
            {},
        ).get(
            "generation_latency_ms",
            0,
        )
        for result in results
    ]

    summary = {
        "benchmark": {
            "total_cases": len(results),
            "known_cases": len(known),
            "multi_section_cases": len(multi),
            "unknown_cases": len(unknown),
        },

        "answer_quality": {
            "correctness_percent": correctness,
            "faithfulness_percent": faithfulness,
            "citation_accuracy_percent": citation_accuracy,
        },

        "multi_section": {
            "fully_correct_percent": percentage(
                multi_correct,
                len(multi),
            ),
            "fully_faithful_percent": percentage(
                multi_faithful,
                len(multi),
            ),
            "fully_cited_percent": percentage(
                multi_citation,
                len(multi),
            ),
        },

        "unknown_handling": {
            "abstention_percent": unknown_abstention_rate,
            "unknown_citation_percent": unknown_citation_rate,
        },

        "latency": {
            "average_judge_latency_ms": round(
                mean(latencies),
                2,
            )
            if latencies
            else 0.0,
            "average_generation_latency_ms": round(
                mean(generation_latencies),
                2,
            )
            if generation_latencies
            else 0.0,
        },

        "failed_multi_section_cases": [
            {
                "id": result["id"],
                "question": result["question"],
                "correctness": result["judge"].get(
                    "correctness"
                ),
                "faithfulness": result["judge"].get(
                    "faithfulness"
                ),
                "citation_accuracy": result["judge"].get(
                    "citation_accuracy"
                ),
                "reason": result["judge"].get(
                    "reason",
                    "",
                ),
            }
            for result in multi
            if result["judge"].get("correctness", 0) < 2
        ],

        "unknown_cases_with_possible_hallucination": [
            {
                "id": result["id"],
                "question": result["question"],
                "answer": result["answer"].get(
                    "answer",
                    "",
                ),
                "abstained": result["answer"].get(
                    "abstained",
                    False,
                ),
                "citations": result["answer"].get(
                    "citations",
                    [],
                ),
                "judge_reason": result["judge"].get(
                    "reason",
                    "",
                ),
            }
            for result in unknown
            if not result["answer"].get(
                "abstained",
                False,
            )
        ],
    }

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    print("========================================")
    print("Answer Evaluation Summary")
    print("========================================")

    print()
    print("Benchmark")
    print(f"  Total:          {len(results)}")
    print(f"  Known:          {len(known)}")
    print(f"  Multi-section:  {len(multi)}")
    print(f"  Unknown:        {len(unknown)}")

    print()
    print("Answer Quality")
    print(
        f"  Correctness:       {correctness}%"
    )
    print(
        f"  Faithfulness:      {faithfulness}%"
    )
    print(
        f"  Citation Accuracy: {citation_accuracy}%"
    )

    print()
    print("Multi-section")
    print(
        f"  Fully Correct: "
        f"{percentage(multi_correct, len(multi))}%"
    )
    print(
        f"  Fully Faithful: "
        f"{percentage(multi_faithful, len(multi))}%"
    )
    print(
        f"  Fully Cited: "
        f"{percentage(multi_citation, len(multi))}%"
    )

    print()
    print("Unknown Handling")
    print(
        f"  Abstention: "
        f"{unknown_abstention_rate}%"
    )
    print(
        f"  With Citations: "
        f"{unknown_citation_rate}%"
    )

    print()
    print("Failed Multi-section Cases")

    if summary["failed_multi_section_cases"]:
        for item in summary["failed_multi_section_cases"]:
            print(
                f"  {item['id']}: "
                f"{item['reason']}"
            )
    else:
        print("  None")

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()