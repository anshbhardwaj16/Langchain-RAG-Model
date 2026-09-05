"""
Document loaders — pulls in local files (PDF/txt/md) AND websites,
then splits everything into chunks ready for embedding.
"""

import os
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

import config


def load_local_files(data_dir: str = config.DATA_DIR) -> List[Document]:
    """Load every .pdf / .txt / .md file in a directory."""
    docs: List[Document] = []

    if not os.path.isdir(data_dir):
        print(f"  ⚠️  Data dir not found: {data_dir}")
        return docs

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        try:
            if fname.lower().endswith(".pdf"):
                loaded = PyPDFLoader(fpath).load()
                docs.extend(loaded)
                print(f"  ✅ Loaded PDF: {fname} ({len(loaded)} pages)")
            elif fname.lower().endswith((".txt", ".md")):
                loaded = TextLoader(fpath, encoding="utf-8").load()
                docs.extend(loaded)
                print(f"  ✅ Loaded text: {fname}")
        except Exception as e:
            print(f"  ❌ Failed to load {fname}: {e}")

    return docs


def load_website(url: str) -> List[Document]:
    """Scrape a single webpage's main text content into a Document."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Research Agent Bot)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Strip noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Collapse excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        title = soup.title.string.strip() if soup.title and soup.title.string else url

        print(f"  ✅ Scraped: {title} ({len(clean_text)} chars)")

        return [Document(page_content=clean_text, metadata={"source": url, "title": title})]

    except Exception as e:
        print(f"  ❌ Failed to scrape {url}: {e}")
        return []


def load_websites(urls: List[str]) -> List[Document]:
    docs: List[Document] = []
    for url in urls:
        docs.extend(load_website(url))
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"  📄 Split {len(docs)} documents into {len(chunks)} chunks")
    return chunks


def load_all(urls: List[str] = None) -> List[Document]:
    """Convenience: load local files + optional websites, then chunk."""
    print("📥 Loading local files...")
    docs = load_local_files()

    if urls:
        print("🌐 Scraping websites...")
        docs.extend(load_websites(urls))

    if not docs:
        print("⚠️  No documents loaded. Add files to ./data or pass URLs.")
        return []

    return chunk_documents(docs)
