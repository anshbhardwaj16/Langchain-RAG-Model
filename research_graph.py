"""
Phase 3: a first LangGraph workflow.

This file intentionally starts small. Phase 1 already shows a fixed RAG chain,
and Phase 2 already shows an LLM tool-calling agent. This first graph shows the
new idea LangGraph adds: the developer controls the workflow between steps.

Performance note
----------------
The graph makes four LLM calls in a row, so wall-clock time is roughly:

    sum over nodes of (prompt tokens read + tokens written) + per-call overhead

Three things are tuned below for that:
  1. one cached client instead of a new one per node (reuses the connection),
  2. an output budget per node, so intermediate steps stay short,
  3. a cap on how much upstream text gets pasted into a downstream prompt.
"""

import time
from functools import lru_cache
from typing import Iterator, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

import config


class ResearchState(TypedDict):
    """The shared state object passed from node to node."""

    question: str
    research: str
    analysis: str
    critique: str
    final_answer: str


# ---- Tuning knobs ---------------------------------------------------------
# Output budget (num_predict) per node. The first three nodes produce working
# notes that only the next prompt ever reads, so they are kept short. Only the
# writer node produces text a human actually sees, so it gets the real budget.
RESEARCH_TOKENS = 220
ANALYSIS_TOKENS = 260
CRITIQUE_TOKENS = 160
ANSWER_TOKENS = 800

# Upper bound on how many characters of an earlier node's output are pasted
# into a later node's prompt. Without this the writer prompt grows with the
# sum of all three previous nodes and prompt processing dominates the run.
MAX_CARRY_CHARS = 1500


@lru_cache(maxsize=None)
def get_llm(num_predict: int = ANSWER_TOKENS) -> ChatOllama:
    """
    Return a cached ChatOllama client for a given output budget.

    Building the client is not free: each ChatOllama owns an HTTP client, so
    constructing one per node meant a fresh connection for all four calls.
    Caching keeps one client per budget and lets the connection be reused.

    num_ctx and keep_alive are identical across every cached client on purpose
    -- Ollama reloads the model when a *load* parameter like num_ctx changes,
    while num_predict is only a sampling parameter and is free to vary.
    """
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
        num_predict=num_predict,
        num_ctx=config.OLLAMA_NUM_CTX,
        keep_alive=config.OLLAMA_KEEP_ALIVE,
    )


def warm_up() -> None:
    """
    Ask Ollama to load the model before the first real question.

    The first call of a session pays for reading the weights off disk. Doing it
    up front means that cost lands during startup instead of inside the graph.
    """
    try:
        get_llm(1).invoke([HumanMessage(content="hi")])
    except Exception:
        # A cold-start optimisation must never break the actual run.
        pass


def _content(response) -> str:
    """ChatOllama returns a message object; the graph state stores plain text."""
    return response.content if hasattr(response, "content") else str(response)


def _preview(value: str, limit: int = 220) -> str:
    value = (value or "").replace("\n", " ").strip()
    return value[:limit] + ("..." if len(value) > limit else "")


def _clip(value: str, limit: int = MAX_CARRY_CHARS) -> str:
    """Bound the upstream text a downstream node has to read."""
    value = (value or "").strip()
    if len(value) <= limit:
        return value
    # Cut on a line boundary so the truncated notes still read as a list.
    return value[:limit].rsplit("\n", 1)[0] + "\n[...truncated...]"


def log_node(name: str, before: ResearchState, after: dict, elapsed: float = 0.0) -> None:
    print(f"\n[graph] node: {name} ({elapsed:.1f}s)")
    print(f"[graph] input question: {_preview(before.get('question', ''))}")
    for key, value in after.items():
        print(f"[graph] output {key}: {_preview(value)}")


def _call(
    name: str,
    messages: list[BaseMessage],
    state_key: str,
    num_predict: int,
    state: ResearchState,
) -> dict:
    """Run one node's LLM call, time it, log it, and return the state update."""
    started = time.perf_counter()
    response = get_llm(num_predict).invoke(messages)
    update = {state_key: _content(response)}
    log_node(name, state, update, time.perf_counter() - started)
    return update


# ---- Node prompts ---------------------------------------------------------
# Each node keeps its own prompt. They are split out from the node functions so
# the writer node can be run either buffered or streamed from the same prompt.


def _research_messages(state: ResearchState) -> list[BaseMessage]:
    return [
        SystemMessage(
            content=(
                "You are the researcher node in a learning LangGraph workflow. "
                "List the key facts, unknowns, and assumptions needed to answer "
                "the user's question. Answer as terse bullet points, no prose, "
                "no preamble, at most 10 bullets."
            )
        ),
        HumanMessage(content=state["question"]),
    ]


def _analysis_messages(state: ResearchState) -> list[BaseMessage]:
    return [
        SystemMessage(
            content=(
                "You are the analyst node. Use the research notes to explain "
                "the answer structure, tradeoffs, and reasoning. Answer as terse "
                "bullet points, no preamble, at most 8 bullets."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Research notes:\n{_clip(state['research'])}"
            )
        ),
    ]


def _critic_messages(state: ResearchState) -> list[BaseMessage]:
    return [
        SystemMessage(
            content=(
                "You are the critic node. Identify any weakness, missing context, "
                "or overclaim in the analysis. Reply with at most 5 short bullets, "
                "or exactly 'OK' if there is nothing worth fixing. Do not restate "
                "the analysis."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Analysis:\n{_clip(state['analysis'])}"
            )
        ),
    ]


def _writer_messages(state: ResearchState) -> list[BaseMessage]:
    return [
        SystemMessage(
            content=(
                "You are the writer node. Produce a clear final answer. "
                "Use the research, analysis, and critique, but do not mention "
                "internal node names unless the user asked about the workflow. "
                "Start with the answer itself -- no preamble."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['question']}\n\n"
                f"Research:\n{_clip(state['research'])}\n\n"
                f"Analysis:\n{_clip(state['analysis'])}\n\n"
                f"Critique:\n{_clip(state['critique'])}"
            )
        ),
    ]


# ---- Nodes ----------------------------------------------------------------


def research_node(state: ResearchState) -> dict:
    """
    A node is just a function that receives graph state and returns state updates.
    This one asks Ollama to gather initial research from its existing knowledge.
    """
    return _call("research", _research_messages(state), "research", RESEARCH_TOKENS, state)


def analysis_node(state: ResearchState) -> dict:
    """Turn the research notes into a reasoned interpretation."""
    return _call("analysis", _analysis_messages(state), "analysis", ANALYSIS_TOKENS, state)


def critic_node(state: ResearchState) -> dict:
    """
    Review the analysis. In the next step, this node will drive conditional edges.
    For now, the graph follows a simple straight-line workflow.

    The critic reads only the analysis, not the raw research: the analysis
    already summarises the research, so re-sending it just costs prompt tokens.
    """
    return _call("critic", _critic_messages(state), "critique", CRITIQUE_TOKENS, state)


def final_answer_node(state: ResearchState) -> dict:
    """Write the final answer using all state produced by earlier nodes."""
    return _call("final_answer", _writer_messages(state), "final_answer", ANSWER_TOKENS, state)


def final_answer_stream(state: ResearchState) -> Iterator[str]:
    """
    Stream the final answer chunk by chunk.

    Same prompt as final_answer_node, but the caller can show text as soon as
    the first tokens arrive instead of waiting for the whole generation. The
    total work is unchanged; the time to the first visible character is not.
    """
    for chunk in get_llm(ANSWER_TOKENS).stream(_writer_messages(state)):
        text = _content(chunk)
        if text:
            yield text


def build_graph():
    """
    StateGraph wires node functions together.

    START and END are special markers:
      START -> first node
      last node -> END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("researcher", research_node)
    graph.add_node("analyst", analysis_node)
    graph.add_node("critic", critic_node)
    graph.add_node("writer", final_answer_node)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "analyst")
    graph.add_edge("analyst", "critic")
    graph.add_edge("critic", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Compile the graph once. The compiled graph is stateless and reusable."""
    return build_graph()


def ask(question: str) -> ResearchState:
    """Run the graph with an initial state and return the final state."""
    app = get_graph()
    initial_state: ResearchState = {
        "question": question,
        "research": "",
        "analysis": "",
        "critique": "",
        "final_answer": "",
    }
    return app.invoke(initial_state)
