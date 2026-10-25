import os

os.environ.setdefault("MEM0_TELEMETRY", "False")  # mem0 sends usage telemetry by default

from mem0 import AsyncMemory
from qdrant_client import QdrantClient

from app import config

_GATEWAY = {"openai_base_url": config.LITELLM_URL, "api_key": config.LITELLM_KEY}

# mem0 uses the same gateway (chat + embedding) and the same Qdrant server.
# AsyncMemory calls the sync Qdrant client through asyncio.to_thread internally.
_memory = AsyncMemory.from_config(
    {
        # Remember the user, not the documents: answers are already grounded in the knowledge base
        "custom_instructions": (
            "Only store durable facts about the user: name, role, team, preferences and ongoing goals. "
            "Do not store facts taken from documents or from the assistant's answers."
        ),
        "llm": {"provider": "openai", "config": {"model": config.CHAT_MODEL, **_GATEWAY}},
        "embedder": {
            "provider": "openai",
            "config": {"model": config.EMBED_MODEL, "embedding_dims": config.EMBED_DIM, **_GATEWAY},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "client": QdrantClient(url=config.QDRANT_URL),
                "collection_name": "memories",
                "embedding_model_dims": config.EMBED_DIM,
            },
        },
    }
)


async def recall(query: str, user_id: str) -> str:
    hits = (await _memory.search(query, filters={"user_id": user_id}, top_k=5))["results"]
    return "\n".join(f"- {h['memory']}" for h in hits)


async def remember(question: str, answer: str, user_id: str) -> None:
    # mem0 extracts durable facts with an LLM and merges them with what it already knows
    messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    await _memory.add(messages, user_id=user_id)


async def list_all(user_id: str) -> list[dict]:
    return (await _memory.get_all(filters={"user_id": user_id}))["results"]
