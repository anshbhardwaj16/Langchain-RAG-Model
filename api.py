"""HTTP adapter that exposes the local LangGraph workflow to the web UI."""

from typing import Any
import json
import threading
import time

from fastapi import FastAPI, HTTPException, Request
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
    model: str | None = None
    options: dict[str, Any] | None = None


NEWLINE = chr(10)

# Conversation history is repeated into every node prompt, so it is capped well
# below num_ctx to leave room for the question, the working notes and the answer.
HISTORY_MAX_CHARS = 1200
HISTORY_MSG_CHARS = 400


def _trim(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


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


@app.on_event("startup")
def _warm_up_model():
    """
    Trigger the model load in the background at boot.

    The first generation of a session pays for Ollama loading the weights.
    Doing it on a background thread means the server is ready immediately and
    that cost is usually already paid by the time a question arrives.
    """
    threading.Thread(target=research_graph.warm_up, daemon=True).start()


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
def chat(request: ChatRequest, http_request: Request):
    # Extract the last user message for the current query
    question = next(
        (m.get("content", "") for m in reversed(request.messages) if m.get("role") == "user"),
        "",
    ).strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="A user message is required")

    # Build context from conversation history (last 3 exchanges, to keep it manageable).
    #
    # This has to stay bounded. full_input is pasted into all four node prompts,
    # and an assistant turn can be ANSWER_TOKENS long. Left unclipped, a few turns
    # of history overflow num_ctx and Ollama silently truncates the prompt from the
    # front -- dropping the system message, which wrecks a 3B model.
    context_messages = request.messages[-6:] if len(request.messages) > 1 else request.messages

    history: list[str] = []
    budget = HISTORY_MAX_CHARS
    for m in reversed(context_messages[:-1]):  # newest first, so oldest is dropped first
        line = f"{m.get('role', '').title()}: {_trim(m.get('content', ''), HISTORY_MSG_CHARS)}"
        if len(line) > budget:
            break
        history.append(line)
        budget -= len(line)
    context = NEWLINE.join(reversed(history))

    # Pass context + current question to the LangGraph workflow.
    full_input = (
        f"Previous conversation:{NEWLINE}{context}{NEWLINE}{NEWLINE}Current question: {question}"
        if context
        else question
    )

    def _preview(text: str, limit: int = 200) -> str:
        clean = (text or "").replace("\n", " ").strip()
        return clean[:limit] + ("..." if len(clean) > limit else "")

    def event_stream(transport: str):
        started = time.perf_counter()

        # State object compatible with research_graph node signatures.
        state: research_graph.ResearchState = {
            "question": full_input,
            "research": "",
            "analysis": "",
            "critique": "",
            "final_answer": "",
        }

        steps = [
            ("researcher", "Gathering research", research_graph.research_node, "research"),
            ("analyst", "Analyzing findings", research_graph.analysis_node, "analysis"),
            ("critic", "Reviewing gaps", research_graph.critic_node, "critique"),
        ]

        yield {
            "type": "status",
            "status": "started",
            "label": "Preparing workflow",
            "workflow": "START -> researcher -> analyst -> critic -> writer -> END",
            "transport": transport,
        }

        try:
            for idx, (phase, label, fn, state_key) in enumerate(steps, 1):
                yield {
                    "type": "phase",
                    "phase": phase,
                    "step": idx,
                    "status": "running",
                    "label": label,
                }

                t0 = time.perf_counter()
                update = fn(state)
                state.update(update)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                content = str(state.get(state_key, "") or "")

                yield {
                    "type": "reasoning",
                    "phase": phase,
                    "content": content,
                }

                yield {
                    "type": "phase",
                    "phase": phase,
                    "step": idx,
                    "status": "done",
                    "label": label,
                    "elapsed_ms": elapsed_ms,
                    "preview": _preview(content),
                }

            # The writer is the only node whose output a human reads, so it is
            # streamed: the first words reach the UI as soon as the model emits
            # them, instead of after the whole answer has been generated.
            writer_step = len(steps) + 1
            yield {
                "type": "phase",
                "phase": "writer",
                "step": writer_step,
                "status": "running",
                "label": "Composing answer",
            }
            yield {
                "type": "status",
                "status": "streaming_answer",
                "label": "Streaming response",
            }

            t0 = time.perf_counter()
            parts: list[str] = []
            for piece in research_graph.final_answer_stream(state):
                parts.append(piece)
                yield {
                    "type": "message",
                    "message": {"role": "assistant", "content": piece},
                }

            answer = "".join(parts)
            state["final_answer"] = answer

            yield {
                "type": "phase",
                "phase": "writer",
                "step": writer_step,
                "status": "done",
                "label": "Composing answer",
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "preview": _preview(answer),
            }

            graph_details = {
                "workflow": "START -> researcher -> analyst -> critic -> writer -> END",
                "research": state.get("research", ""),
                "analysis": state.get("analysis", ""),
                "critique": state.get("critique", ""),
                "final_answer": answer,
            }
            total_ms = int((time.perf_counter() - started) * 1000)
            yield {
                "type": "done",
                "graph": graph_details,
                "elapsed_ms": total_ms,
                "done": True,
            }
        except Exception as exc:
            yield {
                "type": "error",
                "error": str(exc),
                "done": True,
            }

    def ndjson_stream():
        for payload in event_stream("ndjson"):
            yield json.dumps(payload, ensure_ascii=False) + "\n"

    def sse_stream():
        for payload in event_stream("sse"):
            event_name = payload.get("type", "message")
            data = json.dumps(payload, ensure_ascii=False)
            yield f"event: {event_name}\ndata: {data}\n\n"

    transport = (http_request.query_params.get("transport") or "").lower()
    accept = (http_request.headers.get("accept") or "").lower()
    wants_sse = transport == "sse" or "text/event-stream" in accept

    if wants_sse:
        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
