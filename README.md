# AI Research Agent — Phase 1: RAG Foundation

A **100% free, fully local** RAG (Retrieval Augmented Generation) system.
No API keys. No cloud costs. Everything runs on your machine.

## Stack
- **LLM**: Ollama (local)
- **Embeddings**: sentence-transformers (local, CPU)
- **Vector DB**: ChromaDB (local, persists to disk)
- **Framework**: LangChain

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make sure Ollama is running with a model pulled
ollama pull llama3.1          # or mistral, phi3, etc.
ollama serve                  # if not already running
```

If you use a different model than `llama3.1`, edit `config.py`:
```python
OLLAMA_MODEL = "mistral"   # <- change this
```

### Web search setup (free — Tavily)

The agent's `web_search` tool uses [Tavily](https://tavily.com), which has a
free tier (1,000 searches/month, no credit card) and is much more reliable
than scraping DuckDuckGo.

1. Sign up free at https://tavily.com and grab your API key
2. Create a file named `.env` in the project root (same folder as `main.py`)
3. Add this line to it:
   ```
   TAVILY_API_KEY=your_key_here
   ```
That's it — `config.py` automatically loads `.env` on startup. Without a key,
`web_search` still works gracefully (it just tells the agent web search isn't
configured, and it falls back to `search_documents`).

## Usage

### 1. Add your documents
Drop PDFs, `.txt`, or `.md` files into the `./data` folder.
(A sample file is already there so you can test immediately.)

To also scrape websites, open `main.py` and add URLs to the `WEBSITES` list:
```python
WEBSITES = [
    "https://example.com/some-article",
]
```

### 2. Index everything
```bash
python main.py index
```
This loads your files + websites, splits them into chunks, embeds them,
and saves everything into a local ChromaDB store (`./chroma_db`).

**Adding more sources later?** Just run `python main.py index` again.
It's incremental — it tracks what's already indexed (in `indexed_sources.json`)
and only embeds *new* files/URLs, skipping anything already in the store.
No duplicates, no need to delete anything.

If you ever want to wipe everything and start fresh:
```bash
python main.py index --rebuild
```

### 3. Chat with your documents (Phase 1 — pure RAG)
```bash
python main.py chat
```
Ask questions. The agent retrieves relevant chunks and answers using your
local LLM, with source citations. This ALWAYS retrieves from your documents
first — a fixed pipeline, no decision-making involved.

### 4. Chat with the tool-calling agent (Phase 2 — agent decides)
```bash
python main.py agent
```
This is the upgrade: the LLM sees two tools —
- `search_documents` — your indexed PDFs/notes/websites
- `web_search` — the live internet (via DuckDuckGo, free, no API key)

...and **decides for itself**, per question, which one(s) to call. Ask it
something in your documents and it'll use `search_documents`. Ask it about
today's news and it'll reach for `web_search` instead. Run with `verbose=True`
by default, so you'll see exactly which tool it picks and why in the terminal.

## How It Works (Files)

| File | Purpose |
|---|---|
| `config.py` | All settings — model names, chunk size, paths |
| `loaders.py` | Loads PDFs/text files + scrapes websites, splits into chunks |
| `vectorstore.py` | Embeds chunks and stores/retrieves them via ChromaDB |
| `rag_chain.py` | Phase 1: fixed retrieval → prompt → LLM pipeline |
| `tools.py` | Phase 2: wraps document search + web search as agent-callable tools |
| `agent.py` | Phase 2: builds the tool-calling agent that decides which tool to use |
| `main.py` | CLI entry point (`index`, `chat`, and `agent` commands) |

## Troubleshooting

- **"Connection refused" on Ollama** → run `ollama serve` in another terminal
- **Slow answers** → try a smaller model (`phi3` is fast and lightweight)
- **Re-indexing** → just run `python main.py index` again; it's incremental (see above)
- **Out of memory** → use `phi3` or `mistral:7b-instruct-q4` (quantized, smaller)
- **Agent doesn't call tools / just answers directly** → not all models support tool calling well. `qwen2.5`, `llama3.1`, and `mistral-nemo` are known to work with Ollama's tool-calling API. If your model ignores tools, try `llama3.1` instead.
- **Agent loops or gives a weird answer** → small models occasionally misuse tools. `max_iterations=4` in `agent.py` caps this so it can't loop forever — you can lower it further if needed.
- **"Web search is not configured"** → you haven't set `TAVILY_API_KEY` in `.env` yet (see Web search setup above). The agent will still work, just without live web access.

## What's Next (Phase 3 & 4)

Now that your agent can choose between two tools, we'll extend it with:
- **Phase 3**: LangGraph multi-agent orchestration (Researcher → Analyzer → Summarizer, with explicit state and branching)
- **Phase 4**: FastAPI deployment + hybrid search + reranking
