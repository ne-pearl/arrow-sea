import cvxpy as cp

__all__ = [
    "important_function",
    "data",
    "incidence",
    "plots",
    "solver",
]

from . import data
from . import incidence
from . import plots

solver = cp.HIGHS  # Default solver for optimization problems
