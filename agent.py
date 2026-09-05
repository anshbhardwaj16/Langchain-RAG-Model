"""
Phase 2: a tool-calling LangChain agent.

Difference from Phase 1's rag_chain.py:
  - rag_chain.py: FIXED pipeline. Every question -> retrieve -> answer. No choice involved.
  - agent.py: the LLM sees a list of tools with descriptions and decides, per
    question, which tool(s) to call (if any), reads the results, and can even
    call a second tool before answering. This is what "agent" means in practice —
    the model is making decisions about its own next action, not just following
    a script.
"""

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

import config
import tools as tools_module

SYSTEM_PROMPT = """You are a careful research assistant with two tools available:

1. search_documents - searches the user's own indexed PDFs, notes, and scraped websites
2. web_search - searches the live internet

RULE: You must ALWAYS call search_documents FIRST, for every question, before
considering web_search — even if you think you already know the answer. Only
call web_search if search_documents returns nothing relevant, or if the
question is clearly about something time-sensitive (today's news, current
prices, very recent events) that could not possibly be in indexed documents.

You may call both tools if needed. Always mention which source(s) your
answer came from. If neither tool has relevant information, say so honestly
rather than guessing."""


def build_agent(vectorstore) -> AgentExecutor:
    tools_module.init_tools(vectorstore)
    tools = tools_module.get_tools()

    llm = ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # prints each tool call the agent makes — great for learning
        handle_parsing_errors=True,
        max_iterations=4,      # safety cap so a confused small model can't loop forever
    )
