# Architecture

The project keeps the original Agent architecture but migrates the business domain to urban traffic operations.

Main layers:

- LLM client: mock or OpenAI-compatible Qwen3.
- Planner: emits `AgentPlan` JSON and falls back to rules.
- ToolRegistry: binds tool name, description, input schema, output schema, function, read-only flag, category and safety notes.
- Tools: sample traffic metrics, segments, events, forecasts, signal plans and local sample action records.
- RAG: local traffic knowledge base with citations.
- Workflows: congestion diagnosis, incident response, traffic daily report, signal optimization assistant and traffic knowledge QA.
- Trace and Eval: each Agent run is saved and evaluated.
