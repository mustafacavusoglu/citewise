"""Chat history shown in the UI (per-user conversation list + messages).

Plain sqlite3, called from async endpoints via asyncio.to_thread (queries take milliseconds).

The agent's context (the checkpointer) shrinks when compacted; this table never does: it keeps what the
user saw, sources included. Conversation id = LangGraph thread_id.
"""

import json
import sqlite3
import threading
import time

from app import config

_db = sqlite3.connect(config.DB_PATH, check_same_thread=False)
_db.row_factory = sqlite3.Row
_db.executescript("""
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_by_user ON conversations (user_id, updated_at);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations (id),
    data TEXT NOT NULL
);
""")
_lock = threading.Lock()  # one connection shared across threads


def owner(conversation_id: str) -> str | None:
    with _lock:
        row = _db.execute("SELECT user_id FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    return row["user_id"] if row else None


def _insert(conversation_id: str, message: dict) -> None:
    _db.execute(
        "INSERT INTO messages (conversation_id, data) VALUES (?, ?)",
        (conversation_id, json.dumps(message, ensure_ascii=False)),
    )


def start(conversation_id: str, user_id: str, question: str) -> None:
    """Called as soon as a question arrives, so the conversation shows up in the sidebar immediately."""
    with _lock, _db:
        _db.execute(
            "INSERT INTO conversations (id, user_id, title, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET updated_at = excluded.updated_at",
            (conversation_id, user_id, question[:80], time.time()),
        )
        _insert(conversation_id, {"role": "user", "content": question})


def finish(conversation_id: str, reply: dict) -> None:
    with _lock, _db:
        _insert(conversation_id, {"role": "assistant", **reply})
        _db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conversation_id))


def conversations(user_id: str) -> list[dict]:
    with _lock:
        rows = _db.execute(
            "SELECT id, title, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC", (user_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def messages(conversation_id: str) -> list[dict]:
    with _lock:
        rows = _db.execute(
            "SELECT data FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
        ).fetchall()
    return [json.loads(row["data"]) for row in rows]
