"""Tests for QC decisions over ground-LST sample windows."""
from __future__ import annotations

import pytest

from lstnet.qc import (
    QC_NO_DATA,
    QC_OK,
    QC_STD_ERROR,
    decide_qc,
)


def test_clean_window_returns_avg_and_ok():
    samples = [290.0, 290.1, 289.9, 290.05]
    avg, flag = decide_qc(samples)
    assert flag == QC_OK
    assert avg == pytest.approx(290.0125)


def test_high_std_flagged():
    samples = [290.0, 295.0]
    _, flag = decide_qc(samples)
    assert flag == QC_STD_ERROR


def test_empty_window_is_no_data():
    avg, flag = decide_qc([])
    assert flag == QC_NO_DATA
    assert avg is None


def test_single_sample_is_no_data():
    # std with ddof=1 needs >= 2 samples
    avg, flag = decide_qc([290.0])
    assert flag == QC_NO_DATA
    assert avg is None


def test_std_just_below_threshold_is_ok():
    # Rule is std > 1.0 -> StdError, so a clearly sub-threshold window is OK.
    samples = [289.9, 290.1]  # std(ddof=1) ~= 0.141
    avg, flag = decide_qc(samples)
    assert flag == QC_OK
    assert avg == pytest.approx(290.0)
