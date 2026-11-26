import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MCP_GOOGLE_SERVER_PATH = str(BASE_DIR / "mcp_servers" / "google_workspace_mcp.py")

MCP_GOOGLE_PORT = int(os.getenv("MCP_GOOGLE_PORT", "8765"))
MCP_GOOGLE_URL = f"http://127.0.0.1:{MCP_GOOGLE_PORT}/mcp"

# RAG & HTTP
MAX_CACHED_THREADS = 20
RETRIEVER_TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
REQUEST_TIMEOUT = 10

# Models (chat model MUST support tool calling)
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# External APIs & Postgres
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
DB_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable",
)
