from __future__ import annotations

from qwen3_agent_sft.rag.schemas import KnowledgeChunk


def chunk_markdown(source: str, content: str) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    current_title = source
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                chunks.append(
                    KnowledgeChunk(
                        id=f"{source}:{len(chunks) + 1}",
                        source=source,
                        title=current_title,
                        text=text,
                    )
                )
            buffer.clear()

    for line in content.splitlines():
        if line.startswith("#"):
            flush()
            current_title = line.lstrip("#").strip() or source
        elif line.strip():
            buffer.append(line.strip())
    flush()
    return chunks
