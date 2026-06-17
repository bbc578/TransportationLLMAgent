from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from qwen3_agent_sft.training.train_lora import _validate_base_model_path


def merge_lora(base_model_path: str, adapter_dir: str, output_dir: str) -> dict[str, str]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install requirements-train.txt first.") from exc
    base_path = _validate_base_model_path(base_model_path)
    adapter_path = Path(adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"LoRA adapter is incomplete: {adapter_path / 'adapter_config.json'} not found. "
            "Run bash scripts/train_qwen3_lora.sh successfully before merging."
        )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_path), device_map="auto", trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_path), trust_remote_code=True, local_files_only=True
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)
    tokenizer.save_pretrained(out)
    report = {"base_model_path": str(base_path), "adapter_dir": adapter_dir, "output_dir": output_dir}
    (out / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", default=os.environ.get("QWEN3_BASE_MODEL_PATH"))
    parser.add_argument("--adapter-dir", default=os.environ.get("QWEN3_LORA_OUTPUT_DIR"))
    parser.add_argument("--output-dir", default=os.environ.get("QWEN3_MERGED_MODEL_DIR"))
    args = parser.parse_args()
    if not args.base_model_path or not args.adapter_dir or not args.output_dir:
        raise SystemExit("Need QWEN3_BASE_MODEL_PATH, QWEN3_LORA_OUTPUT_DIR, QWEN3_MERGED_MODEL_DIR.")
    print(json.dumps(merge_lora(args.base_model_path, args.adapter_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
