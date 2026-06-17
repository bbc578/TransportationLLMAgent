# Interview Notes

This project demonstrates an urban traffic LLM-Agent with Tool Calling, RAG, workflow orchestration, safety guardrails, traceability and evaluation.

Good talking points:

- Traffic operations require structured tools because live facts change.
- Congestion diagnosis combines metrics, events and knowledge.
- Signal optimization is bounded to auxiliary analysis and must not directly control real systems.
- LoRA SFT is used for stable `AgentPlan` JSON, not for memorizing traffic data.
