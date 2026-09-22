import json

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


def generate_answer(question: str, contexts: list[dict]) -> dict:
    context_text = "\n\n".join(
        (
            f"[Source: {c['source']} | "
            f"Section: {c.get('section', '')} | "
            f"Chunk: {c['chunk_index']}]\n"
            f"{c['text']}"
        )
        for c in contexts
    )

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

    content = response.choices[0].message.content or ""

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return {
            "answer": content,
            "citations": [],
            "abstained": False,
        }

    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "abstained": bool(result.get("abstained", False)),
    }