"""Public-API golden tests (Plan 1a final-review Finding #1).

The direct-reader golden in ``tests/test_surfrad.py`` proves the reader
matches legacy SURFRADlst when called with an explicit ``window=4``. That
leaves a gap: the *public* entry point ``compute_ground_lst`` defaulted to
``window_minutes=10`` (±10 min, 21 samples for step=1), which yields a
different LST than the legacy ``range(row-5+step, row+6-step)`` envelope
(±4 min, 9 samples). The default now delegates to each reader's native
legacy windowing via the ``window_minutes=0`` sentinel.

This file is the real gate: the SURFRAD case below calls
``compute_ground_lst`` with DEFAULT args (no explicit window) and must
match ``SURFRADlst('bon','201102121430',0.95)`` ≈ 268.82 K within 0.01 K.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from lstnet import FixedEmissivity, Site, compute_ground_lst
from lstnet.io.surfrad import SurfradReader

FIXTURES = Path(__file__).parent / "fixtures"

# Bondville, 2011-02-12 14:30 UTC — captured from
# methods/site_LST.py::SURFRADlst('bon','201102121430',0.95).
_BON = Site(name="bon", network="SURFRAD", lon=-88.37, lat=40.05)
_GOLDEN_LEGACY_LST_K = 268.82


def test_public_api_surfrad_default_matches_legacy_surfradlst():
    """``compute_ground_lst`` with DEFAULT args must reproduce legacy SURFRADlst.

    Legacy ``range(row-5+step, row+6-step)`` for step=1 → ±4 min / 9 samples.
    The default ``window_minutes`` delegates to the reader's native legacy
    window (sentinel 0); an explicit positive override would diverge.
    """
    result = compute_ground_lst(
        _BON,
        "201102121430",
        FixedEmissivity(0.95),
        SurfradReader(data_dir=FIXTURES),
    )
    assert result.qc_flag == "OK"
    assert result.lst_k == pytest.approx(_GOLDEN_LEGACY_LST_K, abs=0.01)


def test_public_api_explicit_window_diverges_from_legacy():
    """Documents why the default MUST be the legacy window, not ±10 min.

    With ``window_minutes=10`` (the previous default) the public API
    averages 21 samples (step=1) and lands ≈0.14 K away from legacy —
    outside the 0.01 K golden tolerance. This test pins that behaviour so
    a future "tidy up the default to 10" change cannot silently regress.
    """
    result = compute_ground_lst(
        _BON,
        "201102121430",
        FixedEmissivity(0.95),
        SurfradReader(data_dir=FIXTURES),
        window_minutes=10,
    )
    assert result.qc_flag == "OK"
    # Legacy is 268.82; ±10 min averages in extra samples and shifts ~0.14 K.
    assert result.lst_k != pytest.approx(_GOLDEN_LEGACY_LST_K, abs=0.01)
