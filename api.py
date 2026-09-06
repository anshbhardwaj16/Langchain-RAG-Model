"""HTTP adapter that exposes the local LangGraph workflow to the web UI."""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import agent as agent_module
import research_graph
import vectorstore


class ChatRequest(BaseModel):
    messages: list[dict[str, Any]]
    stream: bool = True


app = FastAPI(title="Local RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_vectorstore = None
_agent = None


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = vectorstore.load_vectorstore()
    return _vectorstore


def get_agent():
    global _agent
    if _agent is None:
        store = get_vectorstore()
        if store is not None:
            _agent = agent_module.build_agent(store)
    return _agent


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": config.OLLAMA_MODEL,
        "graph": "START -> researcher -> analyst -> critic -> writer -> END",
        "index_loaded": get_vectorstore() is not None,
    }


@app.get("/api/tags")
def tags():
    # Keeps the UI's existing connection check compatible with this API.
    return {"models": [{"name": config.OLLAMA_MODEL}]}


@app.post("/api/chat")
def chat(request: ChatRequest):
    # Extract the last user message for the current query
    question = next(
        (m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"),
        "",
    ).strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="A user message is required")

    try:
        # Build context from conversation history (last 3 exchanges, to keep it manageable)
        context_messages = request.messages[-6:] if len(request.messages) > 1 else request.messages
        context = "\n".join([
            f"{m.get('role', '').title()}: {m.get('content', '')}"
            for m in context_messages[:-1]  # All except the latest question
        ])
        
        # Pass context + current question to the LangGraph workflow.
        full_input = f"Previous conversation:\n{context}\n\nCurrent question: {question}" if context else question
        result = research_graph.ask(full_input)
        answer = result["final_answer"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    import json
    graph_details = {
        "workflow": "START -> researcher -> analyst -> critic -> writer -> END",
        "research": result.get("research", ""),
        "analysis": result.get("analysis", ""),
        "critique": result.get("critique", ""),
        "final_answer": answer,
    }
    payload = json.dumps({
        "message": {"role": "assistant", "content": answer},
        "graph": graph_details,
        "done": True,
    })
    return StreamingResponse(iter([payload + "\n"]), media_type="application/x-ndjson")
