"""Geometry projection topology optimization (GPTO) reproduction.

The package reproduces the formulation and examples from Smith and
Norato (2020) while keeping paper-faithful and bounded experimental
aggregation modes explicitly separate.
"""

from .cases import available_cases, build_case
from .geometry import AggregationScheme, ProjectionParameters
from .problem import GPTOProblem

__all__ = [
    "AggregationScheme",
    "GPTOProblem",
    "ProjectionParameters",
    "available_cases",
    "build_case",
]

__version__ = "0.1.0"
