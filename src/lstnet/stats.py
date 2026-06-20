"""Validation statistics over matched ground-truth / retrieved-LST pairs.

All functions take a list of :class:`~lstnet.models.ValidationPair` and operate
on the per-pair difference ``retrieved - ground`` (for bias/rmse) or on the
``(ground, retrieved)`` values (for correlation/regression). Pure numpy.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from lstnet.models import ValidationPair, ValidationStats


def _diffs(pairs: Sequence[ValidationPair]) -> np.ndarray:
    return np.array([p.diff for p in pairs], dtype=float)


def _xy(pairs: Sequence[ValidationPair]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([p.ground.lst_k for p in pairs], dtype=float)
    y = np.array([p.retrieved.lst_k for p in pairs], dtype=float)
    return x, y


def bias(pairs: Sequence[ValidationPair]) -> float:
    """Mean of (retrieved - ground)."""
    if not pairs:
        return float("nan")
    return float(np.mean(_diffs(pairs)))


def rmse(pairs: Sequence[ValidationPair]) -> float:
    """Root-mean-square of (retrieved - ground)."""
    if not pairs:
        return float("nan")
    return float(np.sqrt(np.mean(_diffs(pairs) ** 2)))


def pearson_r(pairs: Sequence[ValidationPair]) -> float:
    """Pearson correlation between ground and retrieved LST."""
    if len(pairs) < 2:
        return float("nan")
    x, y = _xy(pairs)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def linear_regression(pairs: Sequence[ValidationPair]) -> tuple[float, float]:
    """Least-squares (slope, intercept) of retrieved vs ground (y = slope·x + intercept)."""
    if len(pairs) < 2:
        return float("nan"), float("nan")
    x, y = _xy(pairs)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def compute_stats(pairs: Sequence[ValidationPair]) -> ValidationStats:
    """Aggregate all stats into a :class:`ValidationStats`."""
    n = len(pairs)
    b = bias(pairs)
    r_rmse = rmse(pairs)
    r_corr = pearson_r(pairs)
    slope, intercept = linear_regression(pairs)
    return ValidationStats(
        n=n, bias=b, rmse=r_rmse, r=r_corr, slope=slope, intercept=intercept
    )
