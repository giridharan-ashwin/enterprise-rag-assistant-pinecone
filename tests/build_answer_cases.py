import json
from pathlib import Path

from tests.eval_utils import (
    expected_targets,
    is_multi_section_case,
    is_unknown_case,
)


ROOT = Path(__file__).resolve().parents[1]

CASES_FILE = ROOT / "tests" / "stress_test_cases.json"
DATA_DIR = ROOT / "data" / "stress_test"
OUTPUT_FILE = ROOT / "tests" / "answer_eval_cases.json"


def load_cases():
    with CASES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_documents():
    documents = {}

    for path in DATA_DIR.glob("*.md"):
        documents[path.name] = path.read_text(encoding="utf-8")

    return documents


def extract_sections(text: str) -> dict[str, str]:
    sections = {}

    current_section = None
    buffer = []

    for line in text.splitlines():
        if line.startswith("#"):
            if current_section is not None:
                sections[current_section] = "\n".join(buffer).strip()

            current_section = line.lstrip("#").strip()
            buffer = []
        else:
            buffer.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(buffer).strip()

    return sections


def normalize_target(target):
    """
    expected_targets() currently returns tuples:
        (source, section)

    Convert them into a consistent dictionary representation.
    """

    if isinstance(target, tuple):
        if len(target) != 2:
            raise ValueError(
                f"Unexpected target tuple: {target}"
            )

        return {
            "source": target[0],
            "section": target[1],
        }

    if isinstance(target, dict):
        return {
            "source": target.get("source", ""),
            "section": target.get("section", ""),
        }

    raise TypeError(
        f"Unsupported target type: {type(target).__name__}"
    )


def main():
    cases = load_cases()
    documents = load_documents()

    section_map = {}

    for source, text in documents.items():
        section_map[source] = extract_sections(text)

    known_single = []
    multi = []
    unknown = []

    for case in cases:
        if is_unknown_case(case):
            unknown.append(case)
        elif is_multi_section_case(case):
            multi.append(case)
        else:
            known_single.append(case)

    # Benchmark composition:
    # 30 known single-section
    # 15 multi-section
    # 15 unknown
    selected = (
        known_single[:30]
        + multi
        + unknown
    )

    output = []

    for case in selected:
        raw_targets = expected_targets(case)

        targets = [
            normalize_target(target)
            for target in raw_targets
        ]

        gold_context = []

        for target in targets:
            source = target["source"]
            section = target["section"]

            section_text = section_map.get(source, {}).get(
                section,
                "",
            )

            gold_context.append(
                {
                    "source": source,
                    "section": section,
                    "text": section_text,
                }
            )

        if is_unknown_case(case):
            case_type = "unknown"
        elif is_multi_section_case(case):
            case_type = "multi"
        else:
            case_type = "known"

        output.append(
            {
                "id": case["id"],
                "type": case_type,
                "question": case.get("question", ""),
                "expected": targets,
                "gold_context": gold_context,
            }
        )

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(f"Created: {OUTPUT_FILE}")
    print(f"Total cases: {len(output)}")
    print(
        f"Known: "
        f"{sum(c['type'] == 'known' for c in output)}"
    )
    print(
        f"Multi-section: "
        f"{sum(c['type'] == 'multi' for c in output)}"
    )
    print(
        f"Unknown: "
        f"{sum(c['type'] == 'unknown' for c in output)}"
    )


if __name__ == "__main__":
    main()