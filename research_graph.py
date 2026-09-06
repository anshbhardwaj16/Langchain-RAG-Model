"""
Phase 3: a first LangGraph workflow.

This file intentionally starts small. Phase 1 already shows a fixed RAG chain,
and Phase 2 already shows an LLM tool-calling agent. This first graph shows the
new idea LangGraph adds: the developer controls the workflow between steps.
"""

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
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


def get_llm():
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
    )


def _content(response) -> str:
    """ChatOllama returns a message object; the graph state stores plain text."""
    return response.content if hasattr(response, "content") else str(response)


def _preview(value: str, limit: int = 220) -> str:
    value = (value or "").replace("\n", " ").strip()
    return value[:limit] + ("..." if len(value) > limit else "")


def log_node(name: str, before: ResearchState, after: dict) -> None:
    print(f"\n[graph] node: {name}")
    print(f"[graph] input question: {_preview(before.get('question', ''))}")
    for key, value in after.items():
        print(f"[graph] output {key}: {_preview(value)}")


def research_node(state: ResearchState) -> dict:
    """
    A node is just a function that receives graph state and returns state updates.
    This one asks Ollama to gather initial research from its existing knowledge.
    """
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the researcher node in a learning LangGraph workflow. "
                    "List the key facts, unknowns, and assumptions needed to answer "
                    "the user's question. Be concise."
                )
            ),
            HumanMessage(content=state["question"]),
        ]
    )
    update = {"research": _content(response)}
    log_node("research", state, update)
    return update


def analysis_node(state: ResearchState) -> dict:
    """Turn the research notes into a reasoned interpretation."""
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the analyst node. Use the research notes to explain "
                    "the answer structure, tradeoffs, and reasoning. Be concise."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Research notes:\n{state['research']}"
                )
            ),
        ]
    )
    update = {"analysis": _content(response)}
    log_node("analysis", state, update)
    return update


def critic_node(state: ResearchState) -> dict:
    """
    Review the analysis. In the next step, this node will drive conditional edges.
    For now, the graph follows a simple straight-line workflow.
    """
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the critic node. Identify any weakness, missing context, "
                    "or overclaim in the analysis. If it is good enough, say so."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Research:\n{state['research']}\n\n"
                    f"Analysis:\n{state['analysis']}"
                )
            ),
        ]
    )
    update = {"critique": _content(response)}
    log_node("critic", state, update)
    return update


def final_answer_node(state: ResearchState) -> dict:
    """Write the final answer using all state produced by earlier nodes."""
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are the writer node. Produce a clear final answer. "
                    "Use the research, analysis, and critique, but do not mention "
                    "internal node names unless the user asked about the workflow."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['question']}\n\n"
                    f"Research:\n{state['research']}\n\n"
                    f"Analysis:\n{state['analysis']}\n\n"
                    f"Critique:\n{state['critique']}"
                )
            ),
        ]
    )
    update = {"final_answer": _content(response)}
    log_node("final_answer", state, update)
    return update


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


def ask(question: str) -> ResearchState:
    """Run the graph with an initial state and return the final state."""
    app = build_graph()
    initial_state: ResearchState = {
        "question": question,
        "research": "",
        "analysis": "",
        "critique": "",
        "final_answer": "",
    }
    return app.invoke(initial_state)
