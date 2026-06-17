from __future__ import annotations

from pathlib import Path


def load_markdown_files(kb_dir: str | Path = "data/knowledge_base") -> dict[str, str]:
    root = Path(kb_dir)
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(root.glob("*.md"))}
