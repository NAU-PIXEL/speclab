"""
Unit tests for drift-aware Planck handling (temp_spread) in emissivity_nem and
emissivity_alpha, and the _bbt_spread inverse.

Behaviour validated on synthetic drifting targets and on the Singularity Black-LT
emission measurement (2026-10-01) before these tests were written; zero drift was
confirmed bit-identical to speclab v0.12.0.
"""
import logging

import numpy as np
import pytest

import speclab
from speclab.functions import _bbt_spread, _planck_spread
from speclab.utils import bbt, rad

WN = np.linspace(220.0, 2000.0, 923)
T_DW = 284.52
T_SAMPLE = 343.6
WIN = (WN >= 500.0) & (WN <= 1700.0)


def synth(eps, spread):
    return eps * _planck_spread(WN, T_SAMPLE, spread) + (1.0 - eps) * rad(WN, T_DW)


# ----------------------------------------------------------------- helpers
class TestBbtSpread:
    def test_zero_spread_is_bbt(self):
        L = rad(WN, T_SAMPLE)
        np.testing.assert_array_equal(_bbt_spread(WN, L, 0.0), bbt(WN, L))

    @pytest.mark.parametrize("temp, spread", [(343.6, 5.0), (343.6, 30.0), (300.0, 47.0), (200.0, 60.0)])
    def test_round_trip_per_channel(self, temp, spread):
        temps = temp + np.linspace(-3.0, 3.0, WN.size)
        back = _bbt_spread(WN, _planck_spread(WN, temps, spread), spread)
        np.testing.assert_allclose(back, temps, atol=1e-8)

    def test_no_solution_is_nan_without_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = _bbt_spread(WN[:4], np.array([-1.0, 0.0, 1e-30, np.nan]), 10.0)
        assert np.isnan(out).all()
        assert "did not converge" not in caplog.text

    def test_per_channel_planck_spread_matches_scalar(self):
        temps = np.full(WN.size, T_SAMPLE)
        np.testing.assert_array_equal(_planck_spread(WN, temps, 20.0), _planck_spread(WN, T_SAMPLE, 20.0))


# ----------------------------------------------------------------- NEM / Alpha
class TestDriftAwareMethods:
    @pytest.mark.parametrize("method", ["nem", "alpha"])
    def test_zero_spread_unchanged(self, method):
        L = synth(0.97, 15.0)
        if method == "nem":
            a = speclab.emissivity_nem(WN, L, inst="nau", downwelling_t=T_DW)
            b = speclab.emissivity_nem(WN, L, inst="nau", downwelling_t=T_DW, temp_spread=0.0)
        else:
            a = speclab.emissivity_alpha(WN, L, downwelling_t=T_DW)
            b = speclab.emissivity_alpha(WN, L, downwelling_t=T_DW, temp_spread=0.0)
        np.testing.assert_array_equal(a["emiss"], b["emiss"])
        assert a["temp"] == b["temp"]
        assert b["temp_spread"] == 0.0

    def test_nem_recovers_drifting_target(self):
        L = synth(1.0, 20.0)
        r = speclab.emissivity_nem(WN, L, inst="nau", max_emiss=1.0, downwelling_t=T_DW, temp_spread=20.0)
        assert r["temp"] == pytest.approx(T_SAMPLE, abs=1e-6)
        np.testing.assert_allclose(r["emiss"][WIN], 1.0, atol=1e-8)

    def test_alpha_recovers_drifting_target(self):
        L = synth(0.98, 20.0)
        r = speclab.emissivity_alpha(WN, L, max_emiss=0.98, downwelling_t=T_DW, temp_spread=20.0)
        assert r["temp"] == pytest.approx(T_SAMPLE, abs=1e-6)
        np.testing.assert_allclose(r["emiss"][WIN], 0.98, atol=1e-8)

    def test_nem_without_spread_shows_slope(self):
        # Ignoring a real drift reproduces the familiar upward NEM slope
        r = speclab.emissivity_nem(WN, synth(1.0, 20.0), inst="nau", max_emiss=1.0, downwelling_t=T_DW)
        lo = r["emiss"][(WN >= 500) & (WN < 700)].mean()
        hi = r["emiss"][(WN >= 1500) & (WN <= 1700)].mean()
        assert hi - lo > 0.002

    def test_methods_agree_when_drift_explains_tilt(self):
        # graybody with the same drift recovers the same spectrum NEM/Alpha see
        L = synth(1.0, 30.0)
        gb = speclab.emissivity_graybody(WN, L, downwelling_t=T_DW, temp_spread=30.0)
        nem = speclab.emissivity_nem(WN, L, inst="nau", max_emiss=1.0, downwelling_t=T_DW, temp_spread=30.0)
        alpha = speclab.emissivity_alpha(WN, L, max_emiss=1.0, downwelling_t=T_DW, temp_spread=30.0)
        for r in (nem, alpha):
            np.testing.assert_allclose(r["emiss"][WIN], gb["emiss"][WIN], atol=1e-6)
            assert r["temp"] == pytest.approx(gb["temp"], abs=1e-4)

    @pytest.mark.parametrize("spread", [(0.0, 5.0), -1.0])
    @pytest.mark.parametrize("method", ["nem", "alpha"])
    def test_invalid_spread(self, method, spread):
        L = synth(0.97, 0.0)
        with pytest.raises(ValueError, match="temp_spread must be a single non-negative value"):
            if method == "nem":
                speclab.emissivity_nem(WN, L, inst="nau", temp_spread=spread)
            else:
                speclab.emissivity_alpha(WN, L, temp_spread=spread)


# ----------------------------------------------------------------- emcal wiring
class TestEmcalSpread:
    @pytest.mark.parametrize("method", ["hullfit", "hullfit_linear"])
    def test_hullfit_rejects_spread(self, method):
        with pytest.raises(ValueError, match="not supported for method"):
            speclab.emcal("/nonexistent", method=method, temp_spread=5.0)

    @pytest.mark.parametrize("method", ["nem", "alpha"])
    def test_bounded_spread_needs_graybody(self, method):
        with pytest.raises(ValueError, match="needs a fit; use method='graybody'"):
            speclab.emcal("/nonexistent", method=method, temp_spread=(0.0, 5.0))
