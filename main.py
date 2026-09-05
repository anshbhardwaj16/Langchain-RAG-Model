"""
AI Research Agent — Phase 1: RAG Foundation

Usage:
    python main.py index               # index only NEW files/URLs since last run
    python main.py index --rebuild     # wipe everything and re-index from scratch
    python main.py chat                # chat with your indexed documents

Add PDFs/.txt/.md files to ./data before running `index`.
Add website URLs to the WEBSITES list below if you want to scrape them too.
"""

import json
import os
import shutil
import sys
from rich.console import Console
from rich.markdown import Markdown

import config
import loaders
import vectorstore
import rag_chain
import agent as agent_module

console = Console()

# --- Add any URLs you want scraped into the knowledge base ---
WEBSITES = [
    "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md",
    # "https://example.com/some-article",
    # "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
]


def load_manifest() -> set:
    """Return the set of source identifiers (file paths / URLs) already indexed."""
    if os.path.exists(config.MANIFEST_PATH):
        with open(config.MANIFEST_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_manifest(sources: set) -> None:
    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(sources), f, indent=2)


def cmd_index(rebuild: bool = False):
    if rebuild:
        console.print("[yellow]🗑️  Rebuild requested — clearing existing index...[/yellow]")
        if os.path.isdir(config.VECTOR_DB_DIR):
            shutil.rmtree(config.VECTOR_DB_DIR)
        if os.path.exists(config.MANIFEST_PATH):
            os.remove(config.MANIFEST_PATH)

    indexed = load_manifest()

    console.print("\n[bold cyan]📥 Checking for new documents...[/bold cyan]\n")

    # Local files are cheap to re-read; filter out ones already indexed by source path
    all_local_docs = loaders.load_local_files()
    new_local_docs = [d for d in all_local_docs if d.metadata.get("source") not in indexed]
    skipped_local = len(all_local_docs) - len(new_local_docs)

    # Websites are NOT re-fetched if already indexed (avoids wasted network calls)
    new_urls = [u for u in WEBSITES if u not in indexed]
    if new_urls:
        console.print("🌐 Scraping new websites...")
    new_web_docs = loaders.load_websites(new_urls) if new_urls else []

    new_docs = new_local_docs + new_web_docs

    if not new_docs:
        console.print("[green]✅ Nothing new — your index is already up to date.[/green]")
        if skipped_local:
            console.print(f"[dim]({skipped_local} local file(s) were already indexed, unchanged)[/dim]\n")
        return

    chunks = loaders.chunk_documents(new_docs)

    vs = vectorstore.load_vectorstore()
    if vs is None:
        vectorstore.build_vectorstore(chunks)
    else:
        vectorstore.add_documents(vs, chunks)

    # Record what we just indexed so next run skips it
    newly_indexed = {d.metadata.get("source") for d in new_docs if d.metadata.get("source")}
    indexed |= newly_indexed
    save_manifest(indexed)

    console.print(
        f"\n[bold green]✅ Indexed {len(new_docs)} new document(s) → {len(chunks)} chunks.[/bold green]"
    )
    if skipped_local:
        console.print(f"[dim]Skipped {skipped_local} local file(s) already indexed.[/dim]")
    console.print("[bold green]Run `python main.py chat` to query it.[/bold green]\n")


def cmd_chat():
    console.print("\n[bold cyan]🤖 Loading vector store...[/bold cyan]")
    vs = vectorstore.load_vectorstore()

    if vs is None:
        console.print("[red]No index found. Run `python main.py index` first.[/red]")
        return

    console.print("[bold green]Ready! Ask questions about your documents (type 'exit' to quit)[/bold green]\n")

    while True:
        question = console.input("[bold yellow]You:[/bold yellow] ")
        if question.strip().lower() in ("exit", "quit", "q"):
            break
        if not question.strip():
            continue

        console.print("[dim]Thinking...[/dim]")
        try:
            answer, sources = rag_chain.ask(vs, question)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("[dim]Is Ollama running? Try: ollama serve[/dim]\n")
            continue

        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(answer))
        if sources:
            console.print(f"\n[dim]📎 Sources: {', '.join(sources)}[/dim]\n")


def cmd_agent():
    """Phase 2: chat with an agent that can choose between your documents and live web search."""
    console.print("\n[bold cyan]🤖 Loading vector store + building agent...[/bold cyan]")
    vs = vectorstore.load_vectorstore()

    if vs is None:
        console.print("[red]No index found. Run `python main.py index` first.[/red]")
        return

    executor = agent_module.build_agent(vs)
    console.print("[bold green]Ready! This agent can search your docs AND the live web.[/bold green]")
    console.print("[dim](type 'exit' to quit)[/dim]\n")

    while True:
        question = console.input("[bold yellow]You:[/bold yellow] ")
        if question.strip().lower() in ("exit", "quit", "q"):
            break
        if not question.strip():
            continue

        try:
            result = executor.invoke({"input": question})
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print(
                "[dim]This could be Ollama not running (try: ollama serve), "
                "or a tool error. Check the message above for details.[/dim]\n"
            )
            continue

        console.print("\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(result["output"]))
        console.print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python main.py [index|chat|agent] [--rebuild][/yellow]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "index":
        cmd_index(rebuild="--rebuild" in sys.argv)
    elif command == "chat":
        cmd_chat()
    elif command == "agent":
        cmd_agent()
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[yellow]Usage: python main.py [index|chat|agent] [--rebuild][/yellow]")
