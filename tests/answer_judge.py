import json
import time
from pathlib import Path

from app.clients import openai_client
from app.config import settings


ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "answer_eval_results.json"
OUTPUT_FILE = ROOT / "results" / "answer_judge_results.json"

JUDGE_MODEL = settings.openai_chat_model


JUDGE_SYSTEM_PROMPT = """You are an evaluator for an enterprise RAG system.

Evaluate the generated answer against the question, expected evidence,
and retrieved context.

You must evaluate four dimensions:

1. correctness
   - Does the answer correctly answer the question?
   - Is it consistent with the expected evidence?

2. faithfulness
   - Are the factual claims in the answer supported by the retrieved context?
   - Penalize unsupported claims or hallucinations.

3. citation_accuracy
   - Do the cited sources/sections actually support the answer?
   - Missing citations for factual claims should reduce this score.

4. abstention
   - For an unknown question, the correct behavior is to abstain.
   - For a known question, the system should not unnecessarily abstain.

Use a 0-2 scale:

0 = incorrect / unsupported
1 = partially correct / partially supported
2 = correct / fully supported

For abstention:

0 = incorrect behavior
1 = correct behavior

Return JSON only:

{
  "correctness": 0,
  "faithfulness": 0,
  "citation_accuracy": 0,
  "abstention": 0,
  "reason": "brief explanation"
}

Do not use outside knowledge.
Judge only using the supplied question, expected evidence,
retrieved context, generated answer, and citations.
"""


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_gold_context(case):
    parts = []

    for item in case.get("gold_context", []):
        parts.append(
            (
                f"[Expected Source: {item.get('source', '')} | "
                f"Section: {item.get('section', '')}]\n"
                f"{item.get('text', '')}"
            )
        )

    return "\n\n".join(parts)


def build_retrieved_context(result):
    parts = []

    for item in result.get("retrieved_contexts", []):
        parts.append(
            (
                f"[Retrieved Source: {item.get('source', '')} | "
                f"Section: {item.get('section', '')} | "
                f"Score: {item.get('score', '')}]\n"
                f"{item.get('text', '')}"
            )
        )

    return "\n\n".join(parts)


def build_citations(result):
    citations = result.get("answer", {}).get(
        "citations",
        [],
    )

    if not citations:
        return "No citations provided."

    return "\n".join(
        (
            f"- Source: {citation.get('source', '')} | "
            f"Section: {citation.get('section', '')} | "
            f"Chunk: {citation.get('chunk_index', '')}"
        )
        for citation in citations
    )


def judge_answer(result, gold_context):
    question = result["question"]

    answer = result.get("answer", {}).get(
        "answer",
        "",
    )

    abstained = result.get("answer", {}).get(
        "abstained",
        False,
    )

    retrieved_context = build_retrieved_context(
        result
    )

    citations = build_citations(result)

    user_prompt = f"""
Question:
{question}

Expected Evidence:
{gold_context}

Retrieved Context:
{retrieved_context}

Generated Answer:
{answer}

System Abstained:
{abstained}

Citations:
{citations}

Evaluate the generated answer using the rubric.
Return JSON only.
"""

    response = openai_client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": JUDGE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content or "{}"

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "correctness": 0,
            "faithfulness": 0,
            "citation_accuracy": 0,
            "abstention": 0,
            "reason": "Judge returned invalid JSON.",
        }


def main():
    results = load_json(INPUT_FILE)

    print(f"Cases to judge: {len(results)}")
    print(f"Judge model: {JUDGE_MODEL}")
    print()

    judged_results = []

    for index, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"[{index}/{len(results)}] "
            f"{result['id']}"
        )

        gold_context = ""

        # Gold context is stored in answer_eval_cases.json,
        # so load it separately below.
        judged_results.append(
            result
        )

    cases_file = ROOT / "tests" / "answer_eval_cases.json"
    cases = load_json(cases_file)

    case_map = {
        case["id"]: case
        for case in cases
    }

    output = []

    for index, result in enumerate(
        judged_results,
        start=1,
    ):
        case = case_map[result["id"]]

        print(
            f"Judging [{index}/{len(judged_results)}] "
            f"{result['id']}"
        )

        start = time.perf_counter()

        judge_result = judge_answer(
            result,
            build_gold_context(case),
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        output.append(
            {
                "id": result["id"],
                "type": result["type"],
                "question": result["question"],
                "answer": result["answer"],
                "judge": judge_result,
                "judge_latency_ms": round(
                    latency_ms,
                    2,
                ),
            }
        )

        print(
            f"  correctness="
            f"{judge_result.get('correctness')} "
            f"faithfulness="
            f"{judge_result.get('faithfulness')} "
            f"citation="
            f"{judge_result.get('citation_accuracy')} "
            f"abstention="
            f"{judge_result.get('abstention')}"
        )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print()
    print("========================================")
    print("Answer Judge Complete")
    print("========================================")
    print(f"Cases judged: {len(output)}")
    print(f"Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()