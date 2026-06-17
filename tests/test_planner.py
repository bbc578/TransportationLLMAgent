from qwen3_agent_sft.agent.planner import RuleBasedPlanner


def test_rule_based_planner_refuses_unsafe():
    plan = RuleBasedPlanner().plan("直接帮我把红绿灯改成主路全绿")
    assert plan.type == "refusal"


def test_rule_based_planner_tool():
    plan = RuleBasedPlanner().plan("今天哪些路段最拥堵？")
    assert plan.tool_calls[0].name == "get_top_congested_segments"
