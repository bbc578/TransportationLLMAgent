from qwen3_agent_sft.rag.pipeline import RAGPipeline


def test_rag_returns_citations():
    result = RAGPipeline().retrieve_knowledge("信控优化建议需要注意什么？")
    assert not result.no_hit
    assert result.citations
