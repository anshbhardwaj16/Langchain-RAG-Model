"""
The actual RAG pipeline: retrieve relevant chunks from the vector store,
stuff them into a prompt, and ask the local Ollama model to answer —
with citations back to source documents.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful research assistant. Answer the question using ONLY
the context below. If the answer isn't in the context, say so clearly —
do not make things up.

Context:
{context}

Question: {question}

Answer (cite source titles/URLs where relevant):"""
)


def get_llm():
    return ChatOllama(
        model=config.OLLAMA_MODEL,
        base_url=config.OLLAMA_BASE_URL,
        temperature=0.2,
    )


def format_docs(docs) -> str:
    """Turn retrieved chunks into a labeled context block."""
    blocks = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source", "unknown")
        title = d.metadata.get("title", source)
        blocks.append(f"[{i}] Source: {title}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def build_rag_chain(vectorstore):
    """Assemble the retrieve -> prompt -> LLM -> parse pipeline."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.TOP_K})
    llm = get_llm()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def ask(vectorstore, question: str):
    """Run a question through the RAG pipeline and print the answer + sources."""
    chain, retriever = build_rag_chain(vectorstore)

    # Get sources separately so we can display them
    docs = retriever.invoke(question)

    answer = chain.invoke(question)

    sources = []
    for d in docs:
        src = d.metadata.get("title", d.metadata.get("source", "unknown"))
        if src not in sources:
            sources.append(src)

    return answer, sources
