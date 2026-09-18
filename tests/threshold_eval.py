from app.services.retrieval import retrieve


TEST_CASES = [
    {
        "question": "What is the remote work policy?",
        "expected_section": "Remote Work",
    },
    {
        "question": "When should employees submit business expenses?",
        "expected_section": "Expenses",
    },
    {
        "question": "What security requirements apply to employees?",
        "expected_section": "Security",
    },
    {
        "question": "How much notice should employees give for planned time off?",
        "expected_section": "Paid Time Off",
    },
    {
        "question": "What is the company's health insurance deductible?",
        "expected_section": None,
    },
    {
        "question": "What is the company's travel policy?",
        "expected_section": None,
    },
]


THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
]


def evaluate_threshold(threshold: float) -> tuple[float, float]:
    known_cases = [
        case for case in TEST_CASES
        if case["expected_section"] is not None
    ]

    unknown_cases = [
        case for case in TEST_CASES
        if case["expected_section"] is None
    ]

    known_correct = 0
    unknown_rejected = 0

    # Known-answer queries
    for case in known_cases:
        results = retrieve(case["question"], top_k=3)

        filtered = [
            result
            for result in results
            if result["score"] >= threshold
        ]

        expected = case["expected_section"]

        if filtered and filtered[0]["section"] == expected:
            known_correct += 1

    # Unknown queries
    for case in unknown_cases:
        results = retrieve(case["question"], top_k=3)

        filtered = [
            result
            for result in results
            if result["score"] >= threshold
        ]

        if not filtered:
            unknown_rejected += 1

    known_recall = (
        known_correct / len(known_cases) * 100
    )

    unknown_rejection = (
        unknown_rejected / len(unknown_cases) * 100
    )

    return known_recall, unknown_rejection


def main():
    print("=" * 72)
    print("RAG SIMILARITY THRESHOLD EXPERIMENT")
    print("=" * 72)

    print(
        f"{'Threshold':<12}"
        f"{'Known Recall':<18}"
        f"{'Unknown Rejection':<20}"
    )

    print("-" * 72)

    for threshold in THRESHOLDS:
        recall, rejection = evaluate_threshold(threshold)

        print(
            f"{threshold:<12.2f}"
            f"{recall:<18.1f}%"
            f"{rejection:<20.1f}%"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
