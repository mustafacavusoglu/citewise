import uuid

from qdrant_client import AsyncQdrantClient, models

from app import config
from app.chunking import chunk_spans
from app.llm import embed_documents, embed_query

DOCUMENTS = "documents"  # full text + metadata; no vectors, read by id only

client = AsyncQdrantClient(url=config.QDRANT_URL)


async def setup() -> None:
    """Run once at startup: create the collections if they are missing."""
    if not await client.collection_exists(config.COLLECTION):
        await client.create_collection(
            config.COLLECTION,
            vectors_config=models.VectorParams(size=config.EMBED_DIM, distance=models.Distance.COSINE),
        )
    if not await client.collection_exists(DOCUMENTS):
        await client.create_collection(DOCUMENTS, vectors_config={})


def _id(key: str) -> str:
    # Deterministic ids: re-ingesting the same source overwrites instead of duplicating
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


async def add_document(text: str, source: str, metadata: dict[str, str]) -> int:
    doc_id = _id(source)
    spans = chunk_spans(text)
    vectors = await embed_documents([text[start:end] for start, end in spans])
    chunks = [
        models.PointStruct(
            id=_id(f"{source}#{i}"),
            vector=vector,
            payload={
                "doc_id": doc_id,
                "source": source,
                "metadata": metadata,
                "text": text[start:end],
                "start": start,  # offsets in the full text → highlighted on the source page
                "end": end,
            },
        )
        for i, ((start, end), vector) in enumerate(zip(spans, vectors, strict=True))
    ]
    await client.upsert(config.COLLECTION, points=chunks)
    document = {"source": source, "metadata": metadata, "text": text}
    await client.upsert(DOCUMENTS, points=[models.PointStruct(id=doc_id, vector={}, payload=document)])
    return len(chunks)


async def get_document(doc_id: str) -> dict | None:
    points = await client.retrieve(DOCUMENTS, [doc_id])
    return points[0].payload if points else None


async def search(query: str, k: int = config.TOP_K) -> list[dict]:
    hits = (await client.query_points(config.COLLECTION, query=await embed_query(query), limit=k)).points
    return [{**h.payload, "score": h.score} for h in hits]
