"""eval-harness: rubric scoring, agreement statistics and preference ranking.

A dependency-free toolkit for the measurement side of LLM evaluation.
"""

from .agreement import (
    cohens_kappa,
    fleiss_kappa,
    krippendorffs_alpha,
    percent_agreement,
    scotts_pi,
)
from .loaders import LoadError, load_annotations, load_comparisons, load_rubric
from .models import Annotation, Comparison, Criterion, ScoredItem, Verdict
from .preference import Standing, bradley_terry, rank, win_counts
from .rubric import Rubric

__version__ = "0.1.0"

__all__ = [
    "Annotation",
    "Comparison",
    "Criterion",
    "LoadError",
    "Rubric",
    "ScoredItem",
    "Standing",
    "Verdict",
    "bradley_terry",
    "cohens_kappa",
    "fleiss_kappa",
    "krippendorffs_alpha",
    "load_annotations",
    "load_comparisons",
    "load_rubric",
    "percent_agreement",
    "rank",
    "scotts_pi",
    "win_counts",
]
