from app.services.retrieval import retrieve


TEST_CASES = [

    # ---------------------------------------------------------
    # DIRECT QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "What is the remote work policy?",
        "expected_section": "Remote Work",
    },
    {
        "question": "How much planned time off notice is recommended?",
        "expected_section": "Paid Time Off",
    },
    {
        "question": "What security requirements apply to employees?",
        "expected_section": "Security",
    },
    {
        "question": "When should business expenses be submitted?",
        "expected_section": "Expenses",
    },
    {
        "question": "How many days per week can employees work remotely?",
        "expected_section": "Remote Work",
    },
    {
        "question": "How many days do employees have to submit expenses?",
        "expected_section": "Expenses",
    },

    # ---------------------------------------------------------
    # PARAPHRASED QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "How many days are employees allowed to work from home?",
        "expected_section": "Remote Work",
    },
    {
        "question": "What amount of advance notice should someone give for vacation?",
        "expected_section": "Paid Time Off",
    },
    {
        "question": "What authentication controls are required for company systems?",
        "expected_section": "Security",
    },
    {
        "question": "What is the deadline for filing a business expense?",
        "expected_section": "Expenses",
    },
    {
        "question": "Can employees work remotely multiple days each week?",
        "expected_section": "Remote Work",
    },
    {
        "question": "When should planned leave be requested?",
        "expected_section": "Paid Time Off",
    },

    # ---------------------------------------------------------
    # KEYWORD-HEAVY QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "remote work days manager approval",
        "expected_section": "Remote Work",
    },
    {
        "question": "PTO advance notice scheduling system",
        "expected_section": "Paid Time Off",
    },
    {
        "question": "MFA credentials confidential information",
        "expected_section": "Security",
    },
    {
        "question": "expense management system thirty days",
        "expected_section": "Expenses",
    },
    {
        "question": "employee remote work collaboration days",
        "expected_section": "Remote Work",
    },
    {
        "question": "security multi-factor authentication credentials",
        "expected_section": "Security",
    },

    # ---------------------------------------------------------
    # SPECIFIC QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "Who needs to approve remote work arrangements?",
        "expected_section": "Remote Work",
    },
    {
        "question": "What system should employees use for planned time off?",
        "expected_section": "Paid Time Off",
    },
    {
        "question": "Are employees allowed to share their company credentials?",
        "expected_section": "Security",
    },
    {
        "question": "What system is used to submit business expenses?",
        "expected_section": "Expenses",
    },

    # ---------------------------------------------------------
    # NEAR-MISS / TRICK QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "Can employees work remotely five days per week?",
        "expected_section": "Remote Work",
    },
    {
        "question": "Can employees share their passwords with coworkers?",
        "expected_section": "Security",
    },
    {
        "question": "Can expenses be submitted more than thirty days after the purchase?",
        "expected_section": "Expenses",
    },
    {
        "question": "Can employees request planned leave without using the scheduling system?",
        "expected_section": "Paid Time Off",
    },

    # ---------------------------------------------------------
    # UNKNOWN QUESTIONS
    # ---------------------------------------------------------

    {
        "question": "What is the company's health insurance deductible?",
        "expected_section": None,
    },
    {
        "question": "What is the company's travel policy?",
        "expected_section": None,
    },
    {
        "question": "How many holidays does the company provide each year?",
        "expected_section": None,
    },
    {
        "question": "What is the company's parental leave policy?",
        "expected_section": None,
    },
    {
        "question": "What retirement plan does the company offer?",
        "expected_section": None,
    },
    {
        "question": "What is the employee salary review process?",
        "expected_section": None,
    },
]


def evaluate():
    top1_correct = 0
    recall_at_3_correct = 0
    unknown_rejected = 0

    known_cases = [
        case
        for case in TEST_CASES
        if case["expected_section"] is not None
    ]

    unknown_cases = [
        case
        for case in TEST_CASES
        if case["expected_section"] is None
    ]

    print("=" * 80)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    for case in TEST_CASES:

        question = case["question"]
        expected = case["expected_section"]

        results = retrieve(
            question,
            top_k=5,
        )

        top_section = (
            results[0]["section"]
            if results
            else None
        )

        sections = [
            result["section"]
            for result in results
        ]

        print("\nQuestion:")
        print(f"  {question}")

        print("Expected:")
        print(f"  {expected}")

        if results:
            print("Results:")

            for result in results:
                print(
                    f"  {result['section']} "
                    f"(similarity={result['score']:.4f}, "
                    f"rerank={result.get('rerank_score', 0):.4f})"
                )
        else:
            print("Results:")
            print("  No relevant results")

        # Known question
        if expected is not None:

            if top_section == expected:
                top1_correct += 1

            if expected in sections:
                recall_at_3_correct += 1

        # Unknown question
        else:

            if not results:
                unknown_rejected += 1

    top1_accuracy = (
        top1_correct / len(known_cases) * 100
        if known_cases
        else 0
    )

    recall_at_3 = (
        recall_at_3_correct / len(known_cases) * 100
        if known_cases
        else 0
    )

    unknown_rejection = (
        unknown_rejected / len(unknown_cases) * 100
        if unknown_cases
        else 0
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"Total questions:     {len(TEST_CASES)}")
    print(f"Known questions:     {len(known_cases)}")
    print(f"Unknown questions:   {len(unknown_cases)}")
    print(f"Top-1 accuracy:      {top1_accuracy:.1f}%")
    print(f"Recall@3:            {recall_at_3:.1f}%")
    print(f"Unknown rejection:   {unknown_rejection:.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    evaluate()