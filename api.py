"""HTTP adapter that exposes the existing RAG pipeline to the web UI."""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
import agent as agent_module
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
    return {"ok": get_vectorstore() is not None, "model": config.OLLAMA_MODEL}


@app.get("/api/tags")
def tags():
    # Keeps the UI's existing connection check compatible with this API.
    return {"models": [{"name": config.OLLAMA_MODEL}]}


@app.post("/api/chat")
def chat(request: ChatRequest):
    question = next(
        (m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"),
        "",
    ).strip()
    if not question:
        raise HTTPException(status_code=400, detail="A user message is required")

    executor = get_agent()
    if executor is None:
        raise HTTPException(status_code=503, detail="No index found. Run: python main.py index")

    try:
        result = executor.invoke({"input": question})
        answer = result["output"]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # The UI already understands Ollama's newline-delimited response format.
    import json
    payload = json.dumps({"message": {"role": "assistant", "content": answer}, "done": True})
    return StreamingResponse(iter([payload + "\n"]), media_type="application/x-ndjson")
