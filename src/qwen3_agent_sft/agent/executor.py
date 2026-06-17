from __future__ import annotations

from typing import Any
from uuid import uuid4

from qwen3_agent_sft.agent.answer_builder import AnswerBuilder
from qwen3_agent_sft.agent.planner import LLMPlanner
from qwen3_agent_sft.agent.safety import check_safety
from qwen3_agent_sft.agent.schemas import AgentPlan, AgentResponse
from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.agent.trace import TraceStore
from qwen3_agent_sft.llm.qwen_client import build_llm_client
from qwen3_agent_sft.workflows.engine import WorkflowEngine


class AgentExecutor:
    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self.planner = LLMPlanner(build_llm_client(), self.registry)
        self.workflow_engine = WorkflowEngine(self.registry)
        self.answer_builder = AnswerBuilder()
        self.trace_store = TraceStore()

    def run(self, user_query: str) -> AgentResponse:
        trace_id = str(uuid4())
        safety = check_safety(user_query)
        plan = (
            AgentPlan(type="refusal", intent="unsafe_operation", safety=safety)
            if safety.should_refuse
            else self.planner.plan(user_query)
        )
        tool_outputs: list[dict[str, Any]] = []
        workflow_steps: list[dict[str, Any]] = []
        citations: list[str] = []

        if plan.type == "tool_call" and not plan.safety.should_refuse:
            for call in plan.tool_calls:
                output = self.registry.execute_tool(call.name, call.arguments)
                tool_outputs.append(output)
                citations.extend(output.get("citations", []))
        elif plan.type == "workflow" and plan.workflow_name:
            tool_outputs, citations, workflow_steps = self.workflow_engine.run(
                plan.workflow_name, plan.arguments
            )

        answer = self.answer_builder.build(user_query, plan, tool_outputs, citations)
        trace = {
            "trace_id": trace_id,
            "user_query": user_query,
            "llm_model": "configured",
            "planner_type": self.planner.__class__.__name__,
            "raw_plan": plan.model_dump(),
            "parsed_plan": plan.model_dump(),
            "tool_calls": [call.model_dump() for call in plan.tool_calls],
            "tool_outputs": tool_outputs,
            "workflow_steps": workflow_steps,
            "retrieved_chunks": [
                chunk for output in tool_outputs for chunk in output.get("chunks", [])
            ],
            "citations": citations,
            "safety_check": safety.model_dump(),
            "final_answer": answer,
            "errors": [],
        }
        self.trace_store.save(trace)
        return AgentResponse(
            trace_id=trace_id,
            answer=answer,
            plan=plan,
            tool_outputs=tool_outputs,
            citations=citations,
            refused=plan.type == "refusal" or plan.safety.should_refuse,
        )
