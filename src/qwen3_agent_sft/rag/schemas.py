from __future__ import annotations

from pydantic import BaseModel, Field


class KnowledgeChunk(BaseModel):
    id: str
    source: str
    title: str
    text: str


class RetrievalResult(BaseModel):
    chunks: list[KnowledgeChunk] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    no_hit: bool = False
