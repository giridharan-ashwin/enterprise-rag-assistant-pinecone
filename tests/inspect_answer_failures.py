import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FILE = ROOT / "results" / "answer_judge_results.json"


def main():
    with FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    for result in results:
        if result["type"] != "multi":
            continue

        judge = result["judge"]

        if judge.get("correctness", 0) >= 2:
            continue

        print("=" * 80)
        print(f"ID: {result['id']}")
        print(f"Question: {result['question']}")

        print()
        print("Generated Answer:")
        print(
            result["answer"].get(
                "answer",
                "",
            )
        )

        print()
        print("Citations:")

        for citation in result["answer"].get(
            "citations",
            [],
        ):
            print(
                f"  - {citation.get('source')} | "
                f"{citation.get('section')}"
            )

        print()
        print("Judge:")
        print(
            f"  Correctness: "
            f"{judge.get('correctness')}"
        )
        print(
            f"  Faithfulness: "
            f"{judge.get('faithfulness')}"
        )
        print(
            f"  Citation: "
            f"{judge.get('citation_accuracy')}"
        )

        print()
        print("Reason:")
        print(
            judge.get(
                "reason",
                "",
            )
        )

        print()


if __name__ == "__main__":
    main()