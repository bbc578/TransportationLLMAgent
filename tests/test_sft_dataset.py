import json
from pathlib import Path

from qwen3_agent_sft.training.build_sft_dataset import build_dataset, load_raw_cases


def test_sft_dataset_builds():
    out = Path("outputs/test_artifacts/sft")
    out.mkdir(parents=True, exist_ok=True)
    cases = load_raw_cases("data/sft/raw_cases.yaml")
    assert len(cases) >= 200
    stats = build_dataset(
        "data/sft/raw_cases.yaml",
        out / "train.jsonl",
        out / "val.jsonl",
        out / "stats.json",
    )
    assert stats["total"] >= 200
    first = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["messages"][2]["role"] == "assistant"
    assert json.loads(first["messages"][2]["content"])["type"]
