# SFT Dataset

`raw_cases.yaml` uses compact traffic case templates. `scripts/prepare_sft_data.py` expands them into at least 250 concrete chat-style SFT records, then writes `train.jsonl`, `val.jsonl`, and `dataset_stats.json`.

The assistant target is always stable JSON. Tool outputs are not included in the target, so the model learns planning, not sample business facts.
