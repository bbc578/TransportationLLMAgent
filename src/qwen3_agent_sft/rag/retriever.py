from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qwen3_agent_sft.rag.schemas import KnowledgeChunk, RetrievalResult


class TfidfRetriever:
    def __init__(self, chunks: list[KnowledgeChunk], min_score: float = 0.12) -> None:
        self.chunks = chunks
        self.min_score = min_score
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        corpus = [f"{c.title}\n{c.text}" for c in chunks] or [""]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not self.chunks:
            return RetrievalResult(no_hit=True)
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        hits = [(self.chunks[i], float(score)) for i, score in ranked if score >= self.min_score]
        return RetrievalResult(
            chunks=[chunk for chunk, _ in hits],
            citations=[f"{chunk.source}#{chunk.title}" for chunk, _ in hits],
            scores=[score for _, score in hits],
            no_hit=not hits,
        )
