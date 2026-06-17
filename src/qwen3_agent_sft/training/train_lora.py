from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import yaml

from qwen3_agent_sft.config import expand_env


def load_config(path: str | Path) -> dict[str, Any]:
    return expand_env(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def train(config: dict[str, Any], qlora: bool = False) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install requirements-train.txt on the remote GPU server first.") from exc

    base_model_path = _validate_base_model_path(config.get("base_model_path"))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    quantization_config = None
    if qlora:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("QLoRA requires bitsandbytes. Install requirements-train.txt first.") from exc
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_path), trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        quantization_config=quantization_config,
    )
    if config.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    dataset = load_dataset(
        "json",
        data_files={"train": config["train_data_path"], "validation": config["val_data_path"]},
    )
    peft_config = LoraConfig(
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=config["target_modules"],
        task_type="CAUSAL_LM",
    )
    train_args = _build_sft_config(SFTConfig, config, output_dir)

    def formatting_func(example: dict[str, Any]) -> str:
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)

    trainer = _build_sft_trainer(
        SFTTrainer=SFTTrainer,
        model=model,
        tokenizer=tokenizer,
        train_args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_args.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _validate_base_model_path(base_model_path: str | None) -> Path:
    if not base_model_path:
        raise FileNotFoundError(
            "QWEN3_BASE_MODEL_PATH is empty. Set it to the local Qwen3 model directory."
        )
    path = Path(base_model_path).expanduser().resolve()
    required_any_tokenizer = ["tokenizer.json", "tokenizer.model", "vocab.json"]
    if not path.exists():
        raise FileNotFoundError(
            f"Base model path does not exist: {path}\n"
            "Run: ls -lh /root/autodl-tmp/agent/model/models\n"
            "If the files are elsewhere, export QWEN3_BASE_MODEL_PATH to the directory containing config.json."
        )
    if not path.is_dir():
        raise NotADirectoryError(f"Base model path is not a directory: {path}")
    missing: list[str] = []
    if not (path / "config.json").exists():
        missing.append("config.json")
    if not any((path / name).exists() for name in required_any_tokenizer):
        missing.append("tokenizer.json or tokenizer.model or vocab.json")
    if not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        missing.append("model weights (*.safetensors or pytorch_model*.bin)")
    if missing:
        raise FileNotFoundError(
            f"Base model directory is incomplete: {path}\n"
            f"Missing: {', '.join(missing)}\n"
            "Set QWEN3_BASE_MODEL_PATH to the directory that directly contains config.json, tokenizer files, and model weights."
        )
    return path


def _build_sft_config(sft_config_cls: type, config: dict[str, Any], output_dir: Path) -> Any:
    signature = inspect.signature(sft_config_cls.__init__)
    accepted = set(signature.parameters)
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(config["learning_rate"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "bf16": bool(config.get("bf16")),
        "fp16": bool(config.get("fp16")),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "report_to": [],
    }
    if "max_seq_length" in accepted:
        kwargs["max_seq_length"] = int(config["max_seq_length"])
    elif "max_length" in accepted:
        kwargs["max_length"] = int(config["max_seq_length"])
    if "eval_strategy" in accepted:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = "steps"
    if "dataset_text_field" in accepted:
        kwargs["dataset_text_field"] = "text"
    return sft_config_cls(**{key: value for key, value in kwargs.items() if key in accepted})


def _build_sft_trainer(
    SFTTrainer: type,
    model: Any,
    tokenizer: Any,
    train_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    peft_config: Any,
    formatting_func: Any,
) -> Any:
    signature = inspect.signature(SFTTrainer.__init__)
    accepted = set(signature.parameters)
    kwargs: dict[str, Any] = {
        "model": model,
        "args": train_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "formatting_func": formatting_func,
    }
    if "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    return SFTTrainer(**{key: value for key, value in kwargs.items() if key in accepted})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--qlora", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    config.setdefault("base_model_path", os.environ.get("QWEN3_BASE_MODEL_PATH"))
    train(config, qlora=args.qlora)


if __name__ == "__main__":
    main()
