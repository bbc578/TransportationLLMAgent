from __future__ import annotations

from pathlib import Path

from qwen3_agent_sft.rag.chunker import chunk_markdown
from qwen3_agent_sft.rag.loader import load_markdown_files
from qwen3_agent_sft.rag.retriever import TfidfRetriever
from qwen3_agent_sft.rag.schemas import RetrievalResult


class RAGPipeline:
    def __init__(self, kb_dir: str | Path = "data/knowledge_base", min_score: float = 0.12) -> None:
        docs = load_markdown_files(kb_dir)
        chunks = [chunk for source, text in docs.items() for chunk in chunk_markdown(source, text)]
        self.retriever = TfidfRetriever(chunks, min_score=min_score)

    def retrieve_knowledge(self, query: str, top_k: int = 5) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k)
