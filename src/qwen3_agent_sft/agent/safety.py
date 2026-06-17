from __future__ import annotations

from qwen3_agent_sft.agent.schemas import SafetyDecision

UNSAFE_KEYWORDS = [
    "直接控制红绿灯",
    "控制红绿灯",
    "下发信号配时",
    "修改真实信号方案",
    "主路全绿",
    "删除交通事件",
    "伪造交通事件",
    "伪造事故",
    "隐瞒事故",
    "绕过人工审核",
    "获取个人隐私",
    "系统命令",
    "访问外部网络",
    "替代交管部门决策",
    "rm -rf",
]


def check_safety(user_query: str) -> SafetyDecision:
    lowered = user_query.lower()
    for keyword in UNSAFE_KEYWORDS:
        if keyword.lower() in lowered:
            return SafetyDecision(should_refuse=True, reason=f"请求包含高风险交通操作: {keyword}")
    return SafetyDecision()
