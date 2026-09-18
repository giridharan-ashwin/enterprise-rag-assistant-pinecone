from app.clients import openai_client
from app.config import settings

SYSTEM_PROMPT = """You are an enterprise knowledge assistant.
Answer using only the supplied context. If the context is insufficient, say so.
Do not invent facts, policies, names, numbers, or sources. Keep answers concise and useful."""

def generate_answer(question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']} | Chunk: {c['chunk_index']}]\n{c['text']}" for c in contexts
    )
    response = openai_client.chat.completions.create(
        model=settings.openai_chat_model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content or ""
