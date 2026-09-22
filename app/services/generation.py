import json
import time
from typing import Any

from app.clients import openai_client
from app.config import settings


SYSTEM_PROMPT = """You are an enterprise knowledge assistant.

Answer the user's question using ONLY the supplied context.

Rules:

1. Do not invent facts, policies, names, numbers, or sources.
2. If the context does not contain enough information to answer the question,
   abstain and clearly say that the knowledge base does not contain enough information.
3. Keep the answer concise and useful.
4. Every factual answer must include citations to the supplied sources.
5. Only cite sources that actually support the answer.
6. For multi-part questions, use all relevant supplied sources.
7. Return valid JSON only.

Required JSON format:

{
  "answer": "your answer",
  "citations": [
    {
      "source": "filename",
      "section": "section name",
      "chunk_index": 0
    }
  ],
  "abstained": false
}

For an insufficient-context answer:

{
  "answer": "I don't have enough information in the knowledge base to answer that question.",
  "citations": [],
  "abstained": true
}
"""


# GPT-4o-mini pricing.
# Keep these values centralized so they can be updated if pricing changes.
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60


def calculate_cost(
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = (
        prompt_tokens / 1_000_000
    ) * INPUT_COST_PER_1M

    output_cost = (
        completion_tokens / 1_000_000
    ) * OUTPUT_COST_PER_1M

    return input_cost + output_cost


def generate_answer(
    question: str,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:

    context_text = "\n\n".join(
        (
            f"[Source: {c['source']} | "
            f"Section: {c.get('section', '')} | "
            f"Chunk: {c['chunk_index']}]\n"
            f"{c['text']}"
        )
        for c in contexts
    )

    start_time = time.perf_counter()

    response = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_text}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    generation_latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    usage = getattr(response, "usage", None)

    prompt_tokens = (
        getattr(usage, "prompt_tokens", None)
        if usage
        else None
    )

    completion_tokens = (
        getattr(usage, "completion_tokens", None)
        if usage
        else None
    )

    total_tokens = (
        getattr(usage, "total_tokens", None)
        if usage
        else None
    )

    estimated_cost_usd = calculate_cost(
        prompt_tokens,
        completion_tokens,
    )

    content = response.choices[0].message.content or ""

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "answer": content,
            "citations": [],
            "abstained": False,
        }

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "abstained": bool(result.get("abstained", False)),
        "generation_latency_ms": generation_latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
    }