"""The generator: clean parent-child data that satisfies the rules exactly, or a spec learned
from real tables. It knows nothing of errors."""
from .distributions import CPT, Depends, Frequencies, Normal, Skewed, Uniform
from .fanout import Empirical, Fixed, Geometric, Poisson, UniformRange
from .fit import FitResult, fit
from .generate import CleanData, generate
from .spec import DatasetSpec

__all__ = [
    "CPT", "Depends", "Frequencies", "Normal", "Skewed", "Uniform",
    "Empirical", "Fixed", "Geometric", "Poisson", "UniformRange",
    "FitResult", "fit", "CleanData", "generate", "DatasetSpec",
]
