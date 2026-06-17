# LoRA Training Design

The SFT target remains `AgentPlan` JSON. The model learns intent routing, tool selection, workflow selection, argument structure and refusal behavior.

It does not learn traffic facts. Metrics, events, forecasts and signal plans are provided by backend tools and RAG at runtime.
