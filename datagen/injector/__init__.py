"""The error injector: errors with a controlled rate, location, mechanism and detectability,
added to clean parent-child data. It needs the schema and the rules, not the generator."""
from .inject import Injection, classify, eligible_attributes, inject_errors
from .spec import ErrorSpec, RowWeight

__all__ = ["ErrorSpec", "RowWeight", "Injection", "inject_errors", "eligible_attributes", "classify"]
