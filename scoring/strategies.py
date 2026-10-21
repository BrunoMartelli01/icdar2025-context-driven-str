"""Named interpretations of the paper/repository context-score conflict."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Strategy = Literal[
    "figure2_independent_max",
    "selected_candidate_paired",
    "repository_paired_threshold",
]
STRATEGIES: tuple[Strategy, ...] = (
    "figure2_independent_max",
    "selected_candidate_paired",
    "repository_paired_threshold",
)


@dataclass(frozen=True)
class ScoreDecision:
    strategy: Strategy
    T1: str
    T2: str
    T3: str
    S1: float
    S3: float
    L1: float
    L3: float
    S: float
    L: float
    C: float
    semantic_winner: Literal["T1", "T3"]
    confidence_winner: Literal["T1", "T3", "independent_maxima"]
    selected_output: str
    needs_fallback: bool
    alpha: float = 0.6
    beta: float = 0.4
    tau: float = 0.8
    tie_policy: str = "ASSUMPTION: T1 wins exact ties"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def decide(
    strategy: Strategy, *, T1: str, T2: str, T3: str,
    S1: float, S3: float, L1: float, L3: float,
) -> ScoreDecision:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    values = (S1, S3, L1, L3)
    if not all(-1.0 <= value <= 1.0 for value in values[:2]) or not all(0 <= value <= 1 for value in values[2:]):
        raise ValueError("Semantic scores must be [-1,1] and lexical scores [0,1]")
    semantic_winner: Literal["T1", "T3"] = "T1" if S1 >= S3 else "T3"
    c1, c3 = 0.6 * S1 + 0.4 * L1, 0.6 * S3 + 0.4 * L3
    if strategy == "figure2_independent_max":
        S, L = max(S1, S3), max(L1, L3)
        confidence_winner: Literal["T1", "T3", "independent_maxima"] = "independent_maxima"
    elif strategy == "selected_candidate_paired":
        S, L = (S1, L1) if semantic_winner == "T1" else (S3, L3)
        confidence_winner = semantic_winner
    else:
        confidence_winner = "T1" if c1 >= c3 else "T3"
        S, L = (S1, L1) if confidence_winner == "T1" else (S3, L3)
    C = 0.6 * S + 0.4 * L
    return ScoreDecision(
        strategy, T1 or "", T2 or "", T3 or "", float(S1), float(S3), float(L1),
        float(L3), float(S), float(L), float(C), semantic_winner, confidence_winner,
        (T1 or "") if semantic_winner == "T1" else (T3 or ""), C < 0.8,
    )

