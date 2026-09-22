from __future__ import annotations

from typing import Any


def normalize(value: Any) -> str:
    return str(value).strip().lower()


def as_set(value: Any) -> set[str]:
    if not value:
        return set()

    if isinstance(value, str):
        return {normalize(value)}

    if isinstance(value, list):
        return {
            normalize(item)
            for item in value
            if item
        }

    return set()


def get_question(
    case: dict[str, Any],
) -> str:
    for key in (
        "question",
        "query",
        "prompt",
    ):
        value = case.get(key)

        if value:
            return str(value).strip()

    raise ValueError(
        f"No question found in case: {case}"
    )


def is_unknown_case(
    case: dict[str, Any],
) -> bool:

    if case.get("known") is False:
        return True

    for key in (
        "type",
        "category",
        "expected_retrieval",
        "classification",
    ):
        value = normalize(
            case.get(key, "")
        )

        if value in {
            "unknown",
            "unanswerable",
            "negative",
            "out_of_scope",
        }:
            return True

    identifier = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    return identifier.startswith("unknown-")


def expected_targets(
    case: dict[str, Any],
) -> list[tuple[str, str]]:
    """
    Return expected retrieval targets as:

        [
            ("source_a.md", "Section A"),
            ("source_b.md", "Section B"),
        ]

    Current multi-section schema:

        "expected": [
            {
                "source": "...",
                "section": "..."
            },
            ...
        ]

    Also supports the older single-target schema.
    """

    expected = case.get("expected")

    if isinstance(expected, list):

        targets = []

        for item in expected:

            if not isinstance(item, dict):
                continue

            source = normalize(
                item.get("source", "")
            )

            section = normalize(
                item.get("section", "")
            )

            if source or section:
                targets.append(
                    (source, section)
                )

        return targets

    source = normalize(
        case.get("expected_source")
        or case.get("source")
        or ""
    )

    section = normalize(
        case.get("expected_section")
        or case.get("section")
        or ""
    )

    if source or section:
        return [
            (source, section)
        ]

    return []


def expected_sources(
    case: dict[str, Any],
) -> set[str]:

    return {
        source
        for source, _ in expected_targets(case)
        if source
    }


def expected_sections(
    case: dict[str, Any],
) -> set[str]:

    return {
        section
        for _, section in expected_targets(case)
        if section
    }


def is_multi_section_case(
    case: dict[str, Any],
) -> bool:

    targets = expected_targets(case)

    if len(targets) > 1:
        return True

    if normalize(
        case.get("type", "")
    ) == "multi":
        return True

    for key in (
        "multi_section",
        "multi_sections",
        "requires_multiple_sections",
        "multi_document",
        "multi_doc",
    ):
        if case.get(key) is True:
            return True

    identifier = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    return identifier.startswith("multi-")


def target_match(
    result: dict[str, Any],
    target: tuple[str, str],
) -> bool:

    expected_source, expected_section = target

    actual_source = normalize(
        result.get("source", "")
    )

    actual_section = normalize(
        result.get("section", "")
    )

    source_match = (
        not expected_source
        or actual_source == expected_source
    )

    section_match = (
        not expected_section
        or actual_section == expected_section
    )

    return (
        source_match
        and section_match
    )


def retrieval_matches_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[bool, bool]:
    """
    Returns:

        top1_match
        recall_at_3_match

    Unknown:
        Correct only when no result survives.

    Known single-target:
        Top-1 / Recall@3 use the expected target.

    Known multi-section:
        Top-1 means the first result is one of the
        required targets.

        Recall@3 requires ALL expected targets
        to appear in the top 3.
    """

    if is_unknown_case(case):

        rejected = (
            len(results) == 0
        )

        return rejected, rejected

    targets = expected_targets(case)

    if not results or not targets:
        return False, False

    multi = is_multi_section_case(case)

    top1_match = any(
        target_match(
            results[0],
            target,
        )
        for target in targets
    )

    top3 = results[:3]

    if multi:

        returned_pairs = {
            (
                normalize(
                    result.get(
                        "source",
                        "",
                    )
                ),
                normalize(
                    result.get(
                        "section",
                        "",
                    )
                ),
            )
            for result in top3
        }

        recall_at_3_match = all(
            target in returned_pairs
            for target in targets
        )

    else:

        recall_at_3_match = any(
            target_match(
                result,
                target,
            )
            for result in top3
            for target in targets
        )

    return (
        top1_match,
        recall_at_3_match,
    )


def multi_section_recall(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> float | None:
    """
    For multi-section cases, calculate the fraction
    of expected source/section pairs retrieved in
    the top 5.

    Example:

        Expected:
            A + B

        Retrieved:
            A + C

        Recall = 0.50
    """

    if not is_multi_section_case(case):
        return None

    targets = expected_targets(case)

    if not targets:
        return None

    retrieved_pairs = {
        (
            normalize(
                result.get(
                    "source",
                    "",
                )
            ),
            normalize(
                result.get(
                    "section",
                    "",
                )
            ),
        )
        for result in results[:5]
    }

    hits = sum(
        target in retrieved_pairs
        for target in targets
    )

    return hits / len(targets)