from __future__ import annotations

from typing import Any


def normalize(value: Any) -> str:
    return str(value).strip().lower()


def is_unknown_case(case: dict[str, Any]) -> bool:
    if case.get("known") is False:
        return True

    for key in (
        "type",
        "category",
        "expected_retrieval",
        "classification",
    ):
        value = normalize(case.get(key, ""))

        if value in {
            "unknown",
            "unanswerable",
            "negative",
            "out_of_scope",
        }:
            return True

    case_id = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    return case_id.startswith("unknown-")


def get_question(case: dict[str, Any]) -> str:
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


def _as_set(value: Any) -> set[str]:
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


def expected_sources(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()

    for key in (
        "expected_sources",
        "expected_source",
        "sources",
        "source",
    ):
        values |= _as_set(case.get(key))

    return values


def expected_sections(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()

    for key in (
        "expected_sections",
        "expected_section",
        "sections",
        "section",
    ):
        values |= _as_set(case.get(key))

    # Support common alternate schemas.
    for key in (
        "required_sections",
        "target_sections",
        "answer_sections",
    ):
        values |= _as_set(case.get(key))

    return values


def is_multi_section_case(case: dict[str, Any]) -> bool:
    sections = expected_sections(case)

    if len(sections) > 1:
        return True

    for key in (
        "multi_section",
        "multi_sections",
        "requires_multiple_sections",
        "multi_document",
        "multi_doc",
    ):
        value = case.get(key)

        if value is True:
            return True

    case_id = normalize(
        case.get("id")
        or case.get("question_id")
        or ""
    )

    if case_id.startswith("multi-"):
        return True

    for key in (
        "type",
        "category",
    ):
        value = normalize(case.get(key, ""))

        if value in {
            "multi-section",
            "multi_section",
            "multi-document",
            "multi_document",
            "multi-hop",
            "multi_hop",
        }:
            return True

    return False


def retrieval_matches_case(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> tuple[bool, bool]:
    """
    Returns:
        top1_match,
        recall_at_k_match
    """

    if is_unknown_case(case):
        rejected = len(results) == 0
        return rejected, rejected

    if not results:
        return False, False

    wanted_sources = expected_sources(case)
    wanted_sections = expected_sections(case)

    top = results[0]

    top_source = normalize(
        top.get("source", "")
    )

    top_section = normalize(
        top.get("section", "")
    )

    top_source_match = (
        not wanted_sources
        or top_source in wanted_sources
    )

    top_section_match = (
        not wanted_sections
        or top_section in wanted_sections
    )

    top1_match = (
        top_source_match
        and top_section_match
    )

    top_k = results[:3]

    returned_sources = {
        normalize(result.get("source", ""))
        for result in top_k
    }

    returned_sections = {
        normalize(result.get("section", ""))
        for result in top_k
    }

    recall_source_match = (
        not wanted_sources
        or bool(returned_sources & wanted_sources)
    )

    recall_section_match = (
        not wanted_sections
        or bool(returned_sections & wanted_sections)
    )

    recall_at_k_match = (
        recall_source_match
        and recall_section_match
    )

    return top1_match, recall_at_k_match


def multi_section_recall(
    case: dict[str, Any],
    results: list[dict[str, Any]],
) -> float | None:
    wanted_sections = expected_sections(case)

    if not is_multi_section_case(case):
        return None

    if not wanted_sections:
        return None

    returned_sections = {
        normalize(result.get("section", ""))
        for result in results[:5]
    }

    hits = (
        returned_sections
        & wanted_sections
    )

    return len(hits) / len(wanted_sections)
