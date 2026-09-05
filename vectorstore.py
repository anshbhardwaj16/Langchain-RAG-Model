"""
Vector store management — local ChromaDB + free local embeddings
(sentence-transformers). No API keys, no cloud, no cost.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config

_embeddings = None


def get_embeddings():
    """Lazy-load the embedding model (downloads once, then cached locally)."""
    global _embeddings
    if _embeddings is None:
        print(f"🔎 Loading embedding model: {config.EMBEDDING_MODEL} (first run downloads ~90MB)")
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Create (or overwrite) the persistent vector store from document chunks."""
    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        persist_directory=config.VECTOR_DB_DIR,
    )
    print(f"💾 Vector store saved to {config.VECTOR_DB_DIR} ({len(chunks)} chunks indexed)")
    return vs


def load_vectorstore() -> Optional[Chroma]:
    """Load an existing persisted vector store, if one exists."""
    import os

    if not os.path.isdir(config.VECTOR_DB_DIR):
        return None

    embeddings = get_embeddings()
    vs = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=config.VECTOR_DB_DIR,
    )
    return vs


def add_documents(vs: Chroma, chunks: List[Document]) -> None:
    """Add new chunks to an existing vector store (incremental indexing)."""
    vs.add_documents(chunks)
    print(f"➕ Added {len(chunks)} new chunks to vector store")
