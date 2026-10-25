import os

# Every model sits behind an OpenAI-compatible LiteLLM gateway (litellm.yaml); the app only knows aliases.
LITELLM_URL = os.getenv("LITELLM_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "chat")
GATE_MODEL = os.getenv("GATE_MODEL", CHAT_MODEL)  # yes/no relevance check: a cheap, fast model is enough
EMBED_MODEL = os.getenv("EMBED_MODEL", "embedding")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1024"))  # Qwen3-Embedding-0.6B
# Prepended to search queries only. Qwen3-Embedding expects it; set to "" for models that don't.
EMBED_QUERY_PREFIX = os.getenv(
    "EMBED_QUERY_PREFIX", "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

# One line describing the knowledge base: drives the relevance gate and the system prompt
KB_TOPIC = os.getenv("KB_TOPIC", "Northwind Labs' employee handbook (leave, remote work, travel, IT, benefits)")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "docs")
DB_PATH = os.getenv("DB_PATH", "app.db")  # chat history + LangGraph checkpoints
INGEST_KEY = os.getenv("INGEST_KEY")  # Bearer token for POST /ingest; unset = ingest disabled

CHUNK_SIZE = 800
TOP_K = 4

COMPACT_MAX_TOKENS = 3000  # once a thread grows past this, older turns are summarised
COMPACT_KEEP_TURNS = 2  # most recent user turns kept verbatim
