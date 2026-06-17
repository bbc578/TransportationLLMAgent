# Agent Engineering Notes

This repository is a sample-data-only LLM-Agent project for urban traffic operation analysis and incident response workflows.

- Local development uses `LLM_PROVIDER=mock` and never requires a GPU.
- Remote GPU servers run Qwen3 LoRA or QLoRA SFT, merge adapters, and serve an OpenAI-compatible endpoint.
- Business truth comes from backend traffic tools and RAG. The model is trained to emit `AgentPlan` JSON, not to memorize traffic facts.
- Signal-control suggestions are decision-support only and must not be treated as real control commands.
- Unsafe requests must be refused before any tool execution.
