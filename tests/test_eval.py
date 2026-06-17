from qwen3_agent_sft.eval.agent_eval import load_cases, run_eval


def test_eval_cases_expand():
    assert len(load_cases()) >= 50


def test_agent_eval_runs():
    metrics = run_eval()
    assert metrics["trace_saved_rate"] == 1
