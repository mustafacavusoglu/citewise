import asyncio
import logging

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app import config, memory, vectorstore
from app.llm import chat, gate_chat

log = logging.getLogger(__name__)

SYSTEM = (
    f"You are an assistant for this knowledge base: {config.KB_TOPIC}. For any question that needs facts, "
    "search it with the search_docs tool first; search again with different queries if the first results "
    "are not enough. Base the answer only on the passages you found, and say so when they don't contain the "
    "answer. Reply in the user's language.\n\n"
    "Citation rule: right after each fact, write the name of the source it came from in square brackets, "
    "EXACTLY as it appears on the 'Source:' line of the search results, e.g. [leave-policy.md]. "
    "Do not shorten it or add a prefix; use separate brackets for separate sources. "
    "Do not use square brackets for anything else."
)

GATE_PROMPT = (
    f"You are the relevance filter in front of an assistant for this knowledge base: {config.KB_TOPIC}. "
    "Judge the user's LAST message together with the earlier messages. relevant=true if it is a question or "
    "request that could be answered from that knowledge base, or a follow-up to an earlier relevant question "
    "(e.g. 'and for part-time staff?'). relevant=false for greetings, thanks, small talk, coding help, general "
    "knowledge and anything else outside the knowledge base.\n"
    'Return JSON only: {"relevant": true} or {"relevant": false}'
)
GATE_SCHEMA = {
    "title": "gate",
    "description": "Is the user's last message about the knowledge base?",
    "type": "object",
    "properties": {"relevant": {"type": "boolean"}},
    "required": ["relevant"],
}
OFF_TOPIC = f"I can only answer questions about {config.KB_TOPIC}. Try asking about something in those documents."


class State(MessagesState):
    user_id: str
    relevant: bool  # gate decision: when False, search/agent/mem0 never run
    memories: str  # long-term facts from mem0 (per user, across threads)
    summary: str  # older turns condensed by compact (per thread)


# content goes to the LLM; the artifact (raw hits) stays on the ToolMessage for source cards and citation checks
@tool(response_format="content_and_artifact")
async def search_docs(query: str) -> tuple[str, list[dict]]:
    """Semantic search over the knowledge base. Returns the most relevant passages with their source names.
    Use it for every question that needs facts from the documents."""
    hits = await vectorstore.search(query)
    content = "\n\n---\n\n".join(
        f"Source: [{h['source']}]\n" + ", ".join(f"{k}: {v}" for k, v in h["metadata"].items()) + f"\n{h['text']}"
        for h in hits
    )
    return content or "No results.", hits


TOOLS = [search_docs]
agent_llm = chat.bind_tools(TOOLS)
# Reasoning models (e.g. DeepSeek in thinking mode) reject a forced tool_choice; JSON mode works everywhere
gate_llm = gate_chat.with_structured_output(GATE_SCHEMA, method="json_mode")


def _question(state: State) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "human")


# To use a dedicated decision/classifier model instead, only this function's body changes
async def gate(state: State) -> dict:
    """Stops off-topic questions before the expensive path (mem0 + search + agent)."""
    recent = [m for m in state["messages"] if m.type in ("human", "ai") and m.content][-5:]
    transcript = "\n".join(f"{m.type}: {m.content[:300]}" for m in recent)
    try:
        relevant = (await gate_llm.ainvoke([SystemMessage(GATE_PROMPT), HumanMessage(transcript)]))["relevant"]
    except Exception:
        # The gate saves cost, it is not a security control: on failure let the question through
        log.exception("gate failed; treating the question as relevant")
        relevant = True
    return {"relevant": relevant}


def reject(state: State) -> dict:
    return {"messages": [AIMessage(OFF_TOPIC)]}


async def recall(state: State) -> dict:
    return {"memories": await memory.recall(_question(state), state["user_id"])}


async def call_model(state: State) -> dict:
    parts = [SYSTEM]
    if state.get("memories"):
        parts.append(f"What you remember about the user:\n{state['memories']}")
    if state.get("summary"):
        parts.append(f"Summary of the earlier conversation:\n{state['summary']}")
    return {"messages": [await agent_llm.ainvoke([SystemMessage("\n\n".join(parts)), *state["messages"]])]}


_background: set[asyncio.Task] = set()  # unreferenced tasks can be garbage-collected mid-flight


def _log_failure(task: asyncio.Task) -> None:
    _background.discard(task)
    if not task.cancelled() and task.exception():
        log.error("mem0 write failed", exc_info=task.exception())


async def remember(state: State) -> dict:
    """A mem0 write takes a few LLM calls, so it runs in the background instead of delaying the answer."""
    # ponytail: a write still running at shutdown is lost; use a job queue (arq/celery) if that matters
    task = asyncio.create_task(memory.remember(_question(state), state["messages"][-1].content, state["user_id"]))
    _background.add(task)
    task.add_done_callback(_log_failure)
    return {}


async def compact(state: dict) -> dict:
    """Once the history is long, condenses everything except the last KEEP_TURNS turns and removes it from state.
    Not a graph node: main.py runs it in the background after the answer is sent, so nobody waits for it."""
    messages = state["messages"]
    turn_starts = [i for i, m in enumerate(messages) if m.type == "human"]
    if (
        count_tokens_approximately(messages) < config.COMPACT_MAX_TOKENS
        or len(turn_starts) <= config.COMPACT_KEEP_TURNS
    ):
        return {}
    # Cut at a turn boundary so a tool call is never separated from its result
    old = messages[: turn_starts[-config.COMPACT_KEEP_TURNS]]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in old if m.type in ("human", "ai") and m.content)
    summary = (
        await chat.ainvoke(
            "Merge the previous summary and the conversation below into one short summary. "
            "Keep the user's goals, the answers given and important details.\n\n"
            f"Previous summary:\n{state.get('summary') or '-'}\n\nConversation:\n{transcript}"
        )
    ).content
    return {"summary": summary, "messages": [RemoveMessage(id=m.id) for m in old]}


# gate -> (off-topic) reject
#      -> (relevant)  recall -> agent <-> tools -> remember
builder = StateGraph(State)
builder.add_node(gate)
builder.add_node(reject)
builder.add_node(recall)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node(remember)
builder.add_edge(START, "gate")
builder.add_conditional_edges("gate", lambda state: "recall" if state["relevant"] else "reject", ["recall", "reject"])
builder.add_edge("reject", END)
builder.add_edge("recall", "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: "remember"})
builder.add_edge("tools", "agent")
builder.add_edge("remember", END)
# Compiled in main.py: the async checkpointer (AsyncSqliteSaver) is opened at startup
