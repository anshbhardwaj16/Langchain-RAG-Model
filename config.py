"""
Central configuration for the AI Research Agent.
Change model names / paths here — nothing else needs editing.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads a local .env file if present (e.g. TAVILY_API_KEY=...)

# ---- LLM (Ollama) ----
# The Phase 2 agent requires Ollama tool-calling support. qwen2.5:3b is
# lightweight and is already suitable for this project; override with
# OLLAMA_MODEL in .env when using another tool-capable model.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Context window sent to Ollama. Ollama reloads the model whenever num_ctx
# changes, so every client in this project uses this one value. Bigger is not
# better: the KV cache grows with it and prompt processing slows down.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

# How long Ollama keeps the weights resident after a request. The Phase 3
# graph makes four calls back to back; without this the model can be unloaded
# between nodes and pay a multi-second reload each time.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

# ---- Embeddings (local, free, CPU-friendly) ----
# ~90MB, fast, good quality for RAG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- Vector Store ----
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "research_agent_docs"

# ---- Manifest: tracks which sources are already indexed, to avoid duplicates ----
MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "indexed_sources.json")

# ---- Chunking ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---- Retrieval ----
TOP_K = 4  # how many chunks to retrieve per query

# ---- Data dirs ----
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
