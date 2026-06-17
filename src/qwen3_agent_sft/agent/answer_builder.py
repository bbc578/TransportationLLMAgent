from __future__ import annotations

from typing import Any

from qwen3_agent_sft.agent.schemas import AgentPlan


class AnswerBuilder:
    def build(
        self,
        user_query: str,
        plan: AgentPlan,
        tool_outputs: list[dict[str, Any]],
        citations: list[str],
    ) -> str:
        if plan.safety.should_refuse or plan.type == "refusal":
            reason = plan.safety.reason or "请求涉及高风险或越权交通操作。"
            return f"我不能协助执行该请求。原因：{reason}\n\nlimitations: 我只能基于 sample traffic data、工具结果和交通知识库回答。"

        lines = ["处理结果："]
        for output in tool_outputs:
            if output.get("message"):
                lines.append(f"- {output['message']}")
            if "record_count" in output:
                lines.append(
                    "- 指标汇总："
                    f"记录数 {output['record_count']}，"
                    f"平均速度 {output.get('avg_speed_mean')}，"
                    f"平均流量 {output.get('flow_mean')}，"
                    f"平均拥堵指数 {output.get('congestion_index_mean')}。"
                )
            if "items" in output:
                lines.append(f"- 命中 {len(output['items'])} 条样例记录。")
            if output.get("segment"):
                segment = output["segment"]
                lines.append(
                    f"- 路段 {segment.get('road_segment_id')}：{segment.get('name')}，"
                    f"区域 {segment.get('region_id')}，车道数 {segment.get('lanes')}。"
                )
            if output.get("record"):
                record = output["record"]
                lines.append(f"- 已生成样例处置记录：{record.get('record_id')}。")
            if output.get("chunks"):
                lines.append("- 知识库依据：" + "；".join(c["text"][:80] for c in output["chunks"][:2]))
        if plan.intent == "signal_optimization_assistant":
            lines.append(
                "安全边界：本建议仅用于辅助分析，不可直接下发真实信号控制策略，需结合仿真验证和人工审核。"
            )
        lines.append("citations: " + ("；".join(citations) if citations else "无"))
        lines.append("limitations: 以上仅基于 sample traffic data 和 sample traffic knowledge base，不代表真实城市交通系统。")
        return "\n".join(lines)
