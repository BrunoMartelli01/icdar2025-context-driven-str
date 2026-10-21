"""Public context-scoring package API."""

from .context_score import ContextScorer
from .lexical import lexical_similarity
from .semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from .strategies import STRATEGIES, ScoreDecision, decide

__all__ = [
    "ContextScorer", "MPNetApproxConfig", "MPNetSemanticSimilarity",
    "STRATEGIES", "ScoreDecision", "decide", "lexical_similarity",
]
