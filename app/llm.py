from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app import config


# Chat, gate and embedding models all come from the same OpenAI-compatible gateway
def _chat(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, base_url=config.LITELLM_URL, api_key=config.LITELLM_KEY, temperature=0)


chat = _chat(config.CHAT_MODEL)
gate_chat = _chat(config.GATE_MODEL)
_client = AsyncOpenAI(base_url=config.LITELLM_URL, api_key=config.LITELLM_KEY)


async def embed_documents(texts: list[str]) -> list[list[float]]:
    response = await _client.embeddings.create(model=config.EMBED_MODEL, input=texts)
    return [d.embedding for d in response.data]


async def embed_query(text: str) -> list[float]:
    return (await embed_documents([config.EMBED_QUERY_PREFIX + text]))[0]
