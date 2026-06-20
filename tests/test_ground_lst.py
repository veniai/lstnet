"""Tests for ground-truth LST physics (lst_from_radiance, SIGMA)."""
from __future__ import annotations

import pytest

from lstnet.ground_lst import SIGMA, lst_from_radiance


def test_sigma_unified_value():
    assert SIGMA == pytest.approx(5.670374e-8)


def test_blackbody_roundtrip():
    # emiss = 1.0: L_up = sigma * T^4, L_down drops out
    T = 300.0
    l_up = SIGMA * T**4
    assert lst_from_radiance(l_up, 0.0, 1.0) == pytest.approx(T, rel=1e-9)


def test_with_emissivity_roundtrip():
    # Construct L_up from a known T + emiss + L_down, then recover T.
    T = 290.0
    emiss = 0.95
    l_down = 300.0
    l_up = emiss * SIGMA * T**4 + (1.0 - emiss) * l_down
    assert lst_from_radiance(l_up, l_down, emiss) == pytest.approx(T, rel=1e-9)


def test_g1_anchor_values():
    # G1 example (l_up=400, l_down=300, emiss=0.95); pins the documented formula.
    expected = ((400.0 - 0.05 * 300.0) / (0.95 * 5.670374e-8)) ** 0.25
    got = lst_from_radiance(400.0, 300.0, 0.95)
    assert got == pytest.approx(expected, rel=1e-12)
    # sanity: a few hundred W/m^2 upwelling must map to an Earth-like ~290K, not >400K
    assert 285.0 < got < 300.0


def test_non_physical_net_raises():
    # l_up <= (1 - emiss) * l_down → would yield a non-real root
    with pytest.raises(ValueError):
        lst_from_radiance(10.0, 300.0, 0.95)  # 10 < 0.05 * 300 = 15


def test_invalid_emissivity_raises():
    with pytest.raises(ValueError):
        lst_from_radiance(400.0, 300.0, 0.0)
    with pytest.raises(ValueError):
        lst_from_radiance(400.0, 300.0, 1.5)
    with pytest.raises(ValueError):
        lst_from_radiance(400.0, 300.0, -0.1)
