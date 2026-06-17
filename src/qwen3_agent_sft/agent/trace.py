from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceStore:
    def __init__(self, trace_dir: str | Path = "outputs/traces") -> None:
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def save(self, trace: dict[str, Any]) -> Path:
        path = self.trace_dir / f"{trace['trace_id']}.json"
        trace["timestamp"] = trace.get("timestamp") or datetime.now().isoformat()
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def list(self) -> list[str]:
        return sorted(path.stem for path in self.trace_dir.glob("*.json"))

    def read(self, trace_id: str) -> dict[str, Any]:
        return json.loads((self.trace_dir / f"{trace_id}.json").read_text(encoding="utf-8"))
