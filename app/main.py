import asyncio
import logging
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import msgspec
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app import config, history, memory, vectorstore
from app.agent import builder, compact
from app.citations import cite

log = logging.getLogger(__name__)

NonEmpty = Annotated[str, msgspec.Meta(min_length=1)]
UserId = Annotated[str, msgspec.Meta(min_length=1, max_length=64)]


class IngestIn(msgspec.Struct):
    text: NonEmpty
    source: NonEmpty  # the name the agent cites, e.g. a relative file path
    metadata: dict[str, str] = {}  # shown on source cards and the source page


class AskIn(msgspec.Struct):
    question: NonEmpty
    user_id: UserId = "default"  # login name: keys the conversation list and mem0 memory
    thread_id: str | None = None  # short-term conversation (checkpointer) key


async def decode[T](request: Request, type_: type[T]) -> T:
    try:
        return msgspec.json.decode(await request.body(), type=type_)
    except msgspec.DecodeError as e:  # malformed JSON and schema errors
        raise HTTPException(422, str(e)) from e


def to_json(data) -> Response:
    return Response(msgspec.json.encode(data), media_type="application/json")


STATIC = Path(__file__).with_name("static")
graph = None  # compiled in lifespan with the async checkpointer
checkpointer: AsyncSqliteSaver | None = None
active: set[str] = set()  # conversations whose answer is being generated (single process, single event loop)
tidying: dict[str, asyncio.Task] = {}  # conversations being compacted/pruned in the background after an answer


async def tidy(thread_id: str) -> None:
    """After an answer, in the background: summarise a long history and drop the thread's old checkpoints.
    A new question on the same conversation waits for this first, so state never has two writers."""
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        state = await graph.aget_state(cfg)
        if update := await compact(state.values):
            cfg = await graph.aupdate_state(cfg, update, as_node="remember")
        else:
            cfg = state.config
        # Each checkpoint is a full copy of all messages and time travel is unused: keeping the latest is enough.
        # ponytail: SqliteSaver has no aprune; if a DeltaChannel state field is added this delete is no longer safe
        latest = (thread_id, cfg["configurable"]["checkpoint_id"])
        async with checkpointer.lock, checkpointer.conn.cursor() as cur:
            await cur.execute("DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id != ?", latest)
            await cur.execute("DELETE FROM writes WHERE thread_id = ? AND checkpoint_id != ?", latest)
            await checkpointer.conn.commit()
    except Exception:
        # Nothing is lost: compaction retries next turn, unpruned checkpoints only take space
        log.exception("compact/prune failed: %s", thread_id)
    finally:
        tidying.pop(thread_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global graph, checkpointer
    await vectorstore.setup()
    # Thread state lives in SQLite: conversations survive a server restart
    async with AsyncSqliteSaver.from_conn_string(config.DB_PATH) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        yield
        await asyncio.gather(*tidying.values())  # finish before the connection closes


app = FastAPI(title="citewise", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/source/{doc_id}", include_in_schema=False)
def source_page(doc_id: uuid.UUID) -> FileResponse:
    return FileResponse(STATIC / "document.html")


@app.post("/ingest")
async def ingest(request: Request) -> Response:
    # Writing to (and overwriting) the knowledge base needs the key; with no key configured it stays closed
    key = request.headers.get("authorization", "").removeprefix("Bearer ").encode()
    if not config.INGEST_KEY or not secrets.compare_digest(key, config.INGEST_KEY.encode()):
        raise HTTPException(401, "Invalid ingest key")
    body = await decode(request, IngestIn)
    count = await vectorstore.add_document(body.text, body.source, body.metadata)
    return to_json({"chunks": count})


@app.get("/search")
async def search(q: str, k: int = 4) -> Response:
    # Retrieval only, no LLM: for inspecting chunking and embedding quality
    return to_json(await vectorstore.search(q, k))


@app.post("/ask")
async def ask(request: Request) -> Response:
    body = await decode(request, AskIn)
    thread_id = body.thread_id or str(uuid.uuid4())
    # ponytail: password-less login; this only stops writing into someone else's chat by accident, it is not auth
    if await asyncio.to_thread(history.owner, thread_id) not in (None, body.user_id):
        raise HTTPException(404, "Conversation not found")
    # Different conversations run concurrently; one conversation never generates two answers at once
    if thread_id in active:
        raise HTTPException(409, "The previous question in this conversation is still being answered")
    active.add(thread_id)
    try:
        if pending := tidying.get(thread_id):
            await pending  # don't read state before the previous turn's compaction finishes
        await asyncio.to_thread(history.start, thread_id, body.user_id, body.question)
        try:
            result = await graph.ainvoke(
                {"messages": [("user", body.question)], "user_id": body.user_id},
                {"configurable": {"thread_id": thread_id}},
            )
        except Exception:
            log.exception("answer failed: %s", thread_id)
            failed = {"answer": "⚠ Could not generate an answer, please try again.", "sources": [], "unverified": []}
            await asyncio.to_thread(history.finish, thread_id, failed)
            raise HTTPException(502, "Could not generate an answer") from None
        answer, sources, unverified = cite(result["messages"])
        reply = {"answer": answer, "sources": sources, "unverified": unverified}
        await asyncio.to_thread(history.finish, thread_id, reply)
    finally:
        active.discard(thread_id)
    tidying[thread_id] = asyncio.create_task(tidy(thread_id))
    return to_json({**reply, "thread_id": thread_id, "relevant": result["relevant"]})


@app.get("/users/{user_id}/conversations")
async def conversations(user_id: str) -> Response:
    rows = await asyncio.to_thread(history.conversations, user_id)
    return to_json([{**row, "pending": row["id"] in active} for row in rows])


@app.get("/users/{user_id}/conversations/{conversation_id}")
async def conversation(user_id: str, conversation_id: str) -> Response:
    if await asyncio.to_thread(history.owner, conversation_id) != user_id:
        raise HTTPException(404, "Conversation not found")
    return to_json(await asyncio.to_thread(history.messages, conversation_id))


@app.get("/users/{user_id}/conversations/{conversation_id}/state")
async def thread_state(user_id: str, conversation_id: str) -> Response:
    # For watching compaction: the summary plus the messages still in agent state
    if await asyncio.to_thread(history.owner, conversation_id) != user_id:
        raise HTTPException(404, "Conversation not found")
    state = (await graph.aget_state({"configurable": {"thread_id": conversation_id}})).values
    messages = [{"role": m.type, "content": m.content} for m in state.get("messages", [])]
    return to_json({"summary": state.get("summary"), "messages": messages})


@app.get("/documents/{doc_id}")
async def document(doc_id: uuid.UUID) -> Response:
    doc = await vectorstore.get_document(str(doc_id))
    if doc is None:
        raise HTTPException(404, "Document not found")
    return to_json(doc)


@app.get("/memories/{user_id}")
async def memories(user_id: str) -> Response:
    return to_json(await memory.list_all(user_id))
