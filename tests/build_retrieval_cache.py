from __future__ import annotations

import json
from pathlib import Path

from app.services.hybrid import bm25_retriever
from tests.hybrid_eval import dense_retrieve
from tests.stress_test import get_question, load_cases


CACHE_FILE = Path("results/retrieval_cache.json")

DENSE_TOP_K = 10
SPARSE_TOP_K = 10


def main() -> None:
    cases = load_cases()

    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {}

    print()
    print("=" * 90)
    print("BUILDING RETRIEVAL CACHE")
    print("=" * 90)
    print(f"Cases: {len(cases)}")
    print()

    for index, case in enumerate(cases):
        case_id = str(
            case.get("id")
            or case.get("question_id")
            or f"Q{index + 1}"
        )

        question = get_question(case)

        print(
            f"[{index + 1}/{len(cases)}] "
            f"{case_id}: {question}"
        )

        dense = dense_retrieve(
            question,
            top_k=DENSE_TOP_K,
        )

        sparse = bm25_retriever.retrieve(
            question,
            top_k=SPARSE_TOP_K,
        )

        cache[case_id] = {
            "question": question,
            "dense": dense,
            "sparse": sparse,
        }

    CACHE_FILE.write_text(
        json.dumps(
            cache,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print(f"Saved cache: {CACHE_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
