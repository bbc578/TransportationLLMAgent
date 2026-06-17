from __future__ import annotations

from typing import Any

from qwen3_agent_sft.rag.pipeline import RAGPipeline

_PIPELINE: RAGPipeline | None = None


def retrieve_knowledge(query: str, top_k: int = 5) -> dict[str, Any]:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RAGPipeline()
    result = _PIPELINE.retrieve_knowledge(query=query, top_k=top_k)
    return result.model_dump()
