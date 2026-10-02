"""
Unit tests for speclab.emissivity_graybody and its drift-averaged Planck helper.

Behaviour codified here was validated on synthetic gray / non-gray targets and on
the Singularity Black-LT emission measurement (2026-10-01) before these tests
were written.
"""
import logging
import os

import numpy as np
import pytest

import speclab
from speclab import emissivity_graybody
from speclab.functions import _planck_spread
from speclab.utils import rad

# Synthetic set-up mirroring the NAU lab measurement: 4 cm-1 data on ~1.9 cm-1
# spacing, warm sample, downwelling at room temperature.
WN = np.linspace(220.0, 2000.0, 923)
T_DW = 284.52          # downwelling temperature (K)
T_SAMPLE = 343.6       # sample temperature (K)
NOISE = 0.05           # radiance noise, mW m-2 sr-1 (cm-1)-1
WINDOW = (500.0, 1700.0)
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "example_data", "WardRocks_igneous1")


def synth(eps, temp=T_SAMPLE, spread=0.0, noise=0.0, seed=0):
    """Radiance of a target with emissivity *eps* (scalar or spectrum)."""
    dw = rad(WN, T_DW)
    L = eps * _planck_spread(WN, temp, spread) + (1.0 - eps) * dw
    if noise:
        L = L + np.random.default_rng(seed).normal(0.0, noise, WN.size)
    return L


def window_mask():
    return (WN >= WINDOW[0]) & (WN <= WINDOW[1])


# ----------------------------------------------------------------- _planck_spread
class TestPlanckSpread:
    def test_zero_spread_is_planck(self):
        np.testing.assert_array_equal(_planck_spread(WN, 330.0, 0.0), rad(WN, 330.0))

    def test_matches_dense_average(self):
        temps = np.linspace(330.0 - 15.0, 330.0 + 15.0, 20001)
        dense = np.trapezoid(rad(WN[:, None], temps[None, :]), temps, axis=1) / 30.0
        np.testing.assert_allclose(_planck_spread(WN, 330.0, 30.0), dense, rtol=1e-8)

    def test_spread_raises_radiance(self):
        # B is convex in T over the mid-IR, so averaging a ramp exceeds B(mean T)
        assert np.all(_planck_spread(WN, 330.0, 20.0) > rad(WN, 330.0))


# ----------------------------------------------------------------- recovery
class TestRecovery:
    @pytest.mark.parametrize("eps, temp", [(0.93, T_SAMPLE), (0.97, T_SAMPLE), (0.995, T_SAMPLE),
                                           (0.95, 373.15)])
    def test_noise_free_exact(self, eps, temp):
        r = emissivity_graybody(WN, synth(eps, temp), downwelling_t=T_DW)
        assert r["eps_gray"] == pytest.approx(eps, abs=1e-6)
        assert r["temp"] == pytest.approx(temp, abs=1e-4)
        np.testing.assert_allclose(r["emiss"], eps, atol=1e-8)
        assert r["resid_rms"] < 1e-8

    def test_noisy_unbiased_and_sigma_honest(self):
        fits = [emissivity_graybody(WN, synth(0.93, noise=NOISE, seed=s), downwelling_t=T_DW)
                for s in range(40)]
        eps = np.array([f["eps_gray"] for f in fits])
        sig = np.mean([f["eps_sigma"] for f in fits])
        scatter = eps.std(ddof=1)
        assert abs(eps.mean() - 0.93) < 4 * scatter / np.sqrt(eps.size)
        # reported sigma within a factor of 2 of the empirical scatter (validated ratio ~0.9)
        assert 0.5 < sig / scatter < 2.0

    def test_downwelling_rad_equivalent_to_temperature(self):
        L = synth(0.93)
        a = emissivity_graybody(WN, L, downwelling_t=T_DW)
        b = emissivity_graybody(WN, L, downwelling_rad=rad(WN, T_DW))
        assert a["eps_gray"] == pytest.approx(b["eps_gray"], abs=1e-9)
        assert a["temp"] == pytest.approx(b["temp"], abs=1e-6)

    def test_ignoring_downwelling_biases_result(self):
        # Downwelling reflected by a 0.93 surface must not be absorbed silently
        r = emissivity_graybody(WN, synth(0.93))
        assert abs(r["eps_gray"] - 0.93) > 0.005

    def test_noise_weighting_noise_free_exact(self):
        r = emissivity_graybody(WN, synth(0.93), downwelling_t=T_DW,
                                weighting="noise", noise=np.full(WN.size, NOISE))
        assert r["eps_gray"] == pytest.approx(0.93, abs=1e-6)
        assert r["weighting"] == "noise"


# ----------------------------------------------------------------- drift
class TestTempSpread:
    def test_fixed_spread_recovers_drifting_target(self):
        r = emissivity_graybody(WN, synth(0.995, spread=20.0), downwelling_t=T_DW, temp_spread=20.0)
        assert r["eps_gray"] == pytest.approx(0.995, abs=1e-6)
        assert r["temp"] == pytest.approx(T_SAMPLE, abs=1e-4)
        assert r["temp_spread"] == 20.0
        assert np.isnan(r["temp_spread_sigma"])

    def test_isothermal_fit_of_drifting_target_reads_low(self):
        # A drift mimics eps < 1: ignoring it lowers the fitted emissivity
        r = emissivity_graybody(WN, synth(0.995, spread=20.0), downwelling_t=T_DW)
        assert r["eps_gray"] < 0.99

    def test_bounded_spread_stays_in_bounds(self):
        r = emissivity_graybody(WN, synth(0.995, spread=20.0, noise=NOISE), downwelling_t=T_DW,
                                temp_spread=(0.0, 50.0))
        assert 0.0 <= r["temp_spread"] <= 50.0
        assert np.isfinite(r["temp_spread_sigma"])

    def test_bounded_spread_on_isothermal_target_chooses_no_drift(self):
        r = emissivity_graybody(WN, synth(0.93), downwelling_t=T_DW, temp_spread=(0.0, 50.0))
        assert r["temp_spread"] == pytest.approx(0.0, abs=0.5)
        assert r["eps_gray"] == pytest.approx(0.93, abs=1e-3)


# ----------------------------------------------------------------- diagnostics
class TestDiagnostics:
    def test_resid_is_emissivity_minus_constant(self):
        r = emissivity_graybody(WN, synth(0.93, noise=NOISE), downwelling_t=T_DW)
        m = window_mask()
        np.testing.assert_allclose(r["resid"][m], r["emiss"][m] - r["eps_gray"], atol=1e-12)

    def test_gray_target_no_structure_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            r = emissivity_graybody(WN, synth(0.93, noise=NOISE), downwelling_t=T_DW)
        assert r["resid_structure"] < 1.0
        assert "not gray" not in caplog.text

    def test_non_gray_target_flagged(self, caplog):
        eps_spec = 0.95 - 0.10 * np.exp(-0.5 * ((WN - 1050.0) / 60.0) ** 2)
        with caplog.at_level(logging.WARNING):
            r = emissivity_graybody(WN, synth(eps_spec, noise=NOISE), downwelling_t=T_DW)
        assert r["resid_structure"] > 1.0
        assert "not gray" in caplog.text

    def test_eps_temp_correlation_reported(self):
        r = emissivity_graybody(WN, synth(0.93, noise=NOISE), downwelling_t=T_DW)
        assert -1.0 <= r["eps_temp_corr"] < -0.9

    def test_co2_range_excluded_from_fit(self):
        r = emissivity_graybody(WN, synth(0.93), downwelling_t=T_DW, co2_range=(620.0, 720.0))
        assert not r["fit_mask"][(WN >= 620.0) & (WN <= 720.0)].any()
        assert r["eps_gray"] == pytest.approx(0.93, abs=1e-6)

    def test_output_keys(self):
        r = emissivity_graybody(WN, synth(0.93), downwelling_t=T_DW)
        expected = {"wn", "data", "emiss", "temp", "rad_bb", "rad_model", "eps_gray", "eps_sigma",
                    "temp_sigma", "eps_temp_corr", "temp_spread", "temp_spread_sigma", "resid",
                    "resid_rms", "resid_structure", "weighting", "fit_mask", "wn_range",
                    "co2_range", "downwelling_t", "downwelling_e", "downwelling_rad", "nfev",
                    "elapsed"}
        assert expected <= set(r)
        np.testing.assert_allclose(r["rad_model"], synth(0.93), rtol=1e-8)


# ----------------------------------------------------------------- errors
class TestErrors:
    @pytest.mark.parametrize("kwargs, match", [
        ({"weighting": "noise"}, "requires a noise spectrum"),
        ({"weighting": "relative"}, "weighting must be None or 'noise'"),
        ({"weighting": "noise", "noise": np.ones(10)}, "noise shape"),
        ({"temp_spread": -1.0}, "temp_spread must be >= 0"),
        ({"temp_spread": (5.0, 2.0)}, "temp_spread bounds"),
        ({"temp_spread": (-1.0, 5.0)}, "temp_spread bounds"),
        ({"wn_range": (1000.0, 1005.0)}, "valid channels"),
    ])
    def test_invalid_arguments(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            emissivity_graybody(WN, synth(0.93), **kwargs)


# ----------------------------------------------------------------- emcal wiring
class TestEmcal:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method 'mmd'"):
            speclab.emcal("/nonexistent", method="mmd")

    def test_temp_spread_rejected_for_hullfit(self):
        with pytest.raises(ValueError, match="not supported for method='hullfit'"):
            speclab.emcal("/nonexistent", method="hullfit", temp_spread=5.0)

    @pytest.mark.skipif(not os.path.isdir(EXAMPLE_DIR), reason="example data not available")
    def test_graybody_runs_on_example_data(self):
        out = speclab.emcal(EXAMPLE_DIR, method="graybody", plot=False, save=False)
        assert out["method"] == "Graybody (constant emissivity)"
        assert out["data"].shape == (len(out["label"]), len(out["xaxis"]))
        for lbl in out["label"]:
            em = out["emiss_full"][lbl]
            assert np.isfinite(em["eps_gray"]) and np.isfinite(em["temp"])
            assert out["sample_temps"][lbl] == em["temp"]
