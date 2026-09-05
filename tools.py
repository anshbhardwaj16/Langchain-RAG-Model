"""
Tools the agent can choose to call.

Phase 1 gave us a fixed pipeline: always retrieve from documents, always answer.
Phase 2 hands control to the LLM itself — it decides, per question, whether to:
  - search_documents: look in the user's own indexed PDFs/notes/websites
  - web_search: search the live internet for anything not in that index

This is the core idea behind "agents": instead of a hardcoded flow, the model
sees a list of available tools (with descriptions) and picks which to call.
"""

import os

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

import config

# The vector store gets injected at runtime by agent.py, since it depends on
# whichever index main.py loaded — tools.py itself stays stateless otherwise.
_vectorstore = None


def init_tools(vectorstore) -> None:
    """Call this once before building the agent, so search_documents has data to search."""
    global _vectorstore
    _vectorstore = vectorstore


@tool
def search_documents(query: str) -> str:
    """Search the user's own indexed knowledge base (PDFs, notes, and scraped
    websites that were added via `python main.py index`). Use this FIRST for
    any question that might be answered by the user's own documents — it's
    faster and more precise than the open web for topics already indexed."""
    if _vectorstore is None:
        return "No document index is loaded."

    retriever = _vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found in the local index."

    blocks = []
    for d in docs:
        title = d.metadata.get("title", d.metadata.get("source", "unknown"))
        blocks.append(f"Source: {title}\n{d.page_content}")

    return "\n\n---\n\n".join(blocks)


_tavily_client = None


def _get_tavily():
    """Lazily build the Tavily client — only once, and only if a key is configured."""
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None
        _tavily_client = TavilySearchResults(max_results=4, tavily_api_key=api_key)
    return _tavily_client


@tool
def web_search(query: str) -> str:
    """Search the live web. Use this for current events, recent facts, or
    anything that would NOT be in the user's own indexed documents — e.g.
    today's news, current prices, or topics never added to the index."""
    tavily = _get_tavily()

    if tavily is None:
        return (
            "Web search is not configured. Get a free API key at "
            "https://tavily.com and set TAVILY_API_KEY in your .env file "
            "to enable live web search. For now, only search_documents is available."
        )

    try:
        results = tavily.invoke({"query": query})
    except Exception as e:
        return (
            f"Web search failed (error: {e}). Answer using only "
            f"search_documents if relevant, or tell the user web search is "
            f"temporarily unavailable."
        )

    if not results:
        return "No web results found for that query."

    blocks = []
    for r in results:
        title = r.get("title") or r.get("url", "result")
        url = r.get("url", "")
        content = r.get("content", "")
        blocks.append(f"Source: {title} ({url})\n{content}")

    return "\n\n---\n\n".join(blocks)


def get_tools():
    """Return the full tool list the agent can call."""
    return [search_documents, web_search]
