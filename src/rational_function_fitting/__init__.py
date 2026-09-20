"""Rational function fitting using exact linear algebra.

This package fits a rational function p(x)/q(x) to sample data by
solving a homogeneous linear system exactly over the fractions module.
"""

from .core import fit_rational_function

__all__ = ["fit_rational_function"]
