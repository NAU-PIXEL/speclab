#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectroscopy processing functions for emissivity calibration, transmission
calibration, spectral mixture analysis, and spectral library utilities.

Provides
--------
emcal                  Emission calibration (NEM / MMD / convex-hull LS fit).
tracal                 Transmission calibration (AutomateFTIR metadata CSV).
refcal                 Reflectance calibration (AutomateFTIR metadata CSV).
sma                    Spectral Mixture Analysis (NNLS / OLS).
summary_sma            Pretty-print concentration summary table for SMA results.
sort_cube              Sort endmember concentration arrays per pixel.
sum_group_conc         Group-sum concentrations and propagate errors.
scan_sample_labels     List sample labels from a folder without loading spectra.
emissivity_nem         NEM with two-stage TB detection and downwelling correction.
emissivity_alpha       Alpha Residuals — mean-BT reference with max-emissivity rescaling.
emissivity_mmd         Min-Max Difference algorithm.
emissivity_hullfit     NNLS temperature unmixing + convex-hull seeding with strict data ≤ model enforcement.
dehyd                  Remove residual water vapour features from emissivity spectra.
insert_plot_gaps       Inject NaN breaks at spectral gaps for plotting.
resample_spectrum      Interpolate spectrum onto a target grid.
save_instrument_grids / load_instrument_grids
                       Instrument xaxis cache (compact .npz).
merge                  Merge two or more output dicts (speclib or emcal) or HDF5 paths.
"""

import logging
import os
import time
from collections.abc import Callable
from datetime import datetime

from . import utils
from .utils import _to_album
from .plot import plot_emcal, plot_tracal, plot_sma
import h5py
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy.optimize import curve_fit, nnls, lsq_linear, minimize, minimize_scalar, NonlinearConstraint
from scipy.integrate import simpson
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import uniform_filter1d, median_filter, gaussian_filter1d


# =============================================================================
# ========================== remove_continuum =================================
# =============================================================================

def _upper_hull_indices(x: np.ndarray, y: np.ndarray) -> list[int]:
    """
    Return the indices of the upper convex hull of the point set (x, y).

    Assumes *x* is strictly monotonically increasing (as a wavelength axis is).
    Uses Andrew's monotone chain algorithm restricted to the upper hull.

    Parameters
    ----------
    x : np.ndarray
        Sorted x-coordinates, shape (n,).
    y : np.ndarray
        Corresponding y-coordinates, shape (n,).

    Returns
    -------
    list[int]
        Indices into *x* / *y* that form the upper hull, ordered left to right.
    """
    hull: list[int] = []
    for i in range(len(x)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            # Cross product of (a→b) × (a→i); non-negative means left turn or
            # collinear — remove b because it lies below the hull edge a→i.
            cross = (x[b] - x[a]) * (y[i] - y[a]) - (y[b] - y[a]) * (x[i] - x[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return hull


def remove_continuum(
    xaxis:    np.ndarray,
    spectra:  np.ndarray,
    wl_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Apply convex-hull continuum removal to reflectance spectra.

    For each spectrum the upper convex hull of (wavelength, reflectance) is
    computed and interpolated to form the continuum baseline.  The returned
    values are reflectance / continuum, confined to [0, 1], where 1 indicates
    the spectrum lies on the continuum (no absorption) and values below 1
    indicate absorption depth relative to the continuum.

    When *wl_range* is given the hull is computed only over that sub-range;
    channels outside the range are returned as 1.0 (on continuum).  This is
    useful for isolating a specific absorption complex without the global
    envelope dominating the baseline.

    Parameters
    ----------
    xaxis : np.ndarray
        Wavelength axis, shape (n_channels,).
    spectra : np.ndarray
        Reflectance spectra, shape (n_channels,) for a single spectrum or
        (n_spectra, n_channels) for a stack.  Values should be in [0, 1].
    wl_range : tuple[float, float] or None
        ``(wl_lo, wl_hi)`` wavelength bounds (same units as *xaxis*) for the
        hull computation.  Channels outside this window are set to 1.0.
        ``None`` applies the hull to the full spectrum (default).

    Returns
    -------
    np.ndarray
        Continuum-removed spectra, same shape as *spectra*, dtype float64.
    """
    arr = np.atleast_2d(np.asarray(spectra, dtype=float))

    mask = ((xaxis >= wl_range[0]) & (xaxis <= wl_range[1])
            if wl_range is not None else np.ones(len(xaxis), dtype=bool))
    x_sub = xaxis[mask]

    result = np.ones_like(arr)   # 1.0 outside the range — on the continuum
    if x_sub.size >= 2:
        for i, row in enumerate(arr):
            y_sub          = row[mask]
            hull_idx       = _upper_hull_indices(x_sub, y_sub)
            continuum      = np.interp(x_sub, x_sub[hull_idx], y_sub[hull_idx])
            result[i, mask] = np.clip(y_sub / np.maximum(continuum, 1e-10), 0.0, 1.0)

    return result.squeeze() if np.ndim(spectra) == 1 else result


# =============================================================================
# ========================== band_parameters ==================================
# =============================================================================

def band_parameters(
    xaxis:    np.ndarray,
    spectrum: np.ndarray,
    wl_range: tuple[float, float],
) -> dict[str, float] | None:
    """
    Compute band parameters for a single absorption feature.

    Continuum removal is applied over *wl_range* using the convex-hull method
    (see :func:`remove_continuum`).  All parameters are derived from the
    continuum-removed (CR) reflectance within that window.

    Returns ``None`` when the feature is absent or the window covers fewer
    than three spectral channels.

    Parameters
    ----------
    xaxis : np.ndarray
        Wavelength axis, shape (n_channels,).  Must be monotonically increasing.
    spectrum : np.ndarray
        Single reflectance spectrum, shape (n_channels,).  Values in [0, 1].
    wl_range : tuple[float, float]
        ``(wl_lo, wl_hi)`` shoulder-to-shoulder window (same units as *xaxis*)
        used for continuum removal.

    Returns
    -------
    dict[str, float] or None
        Keys: ``wl_center``, ``wl_min``, ``band_depth``, ``fwhm``,
        ``base_width``, ``band_area``, ``band_area_ratio``,
        ``asymmetry_hw``, ``asymmetry_centroid``.

        ``wl_center``
            Spectral centroid: absorption-weighted mean wavelength.
        ``wl_min``
            Wavelength of the band minimum (lowest CR reflectance point).
        ``band_depth``
            Fractional depth: ``1 − CR_min``, in [0, 1].
        ``fwhm``
            Full-width at half-maximum, interpolated to the half-depth level
            on each side of the minimum.
        ``base_width``
            Band width at 10 % of the maximum depth, using the same
            interpolation as *fwhm*.  Feature-intrinsic; independent of the
            shoulder window.
        ``band_area``
            Integral of ``(1 − CR)`` over the 10 %-threshold extent, in the
            same units as *xaxis*.
        ``band_area_ratio``
            ``band_area / (band_depth × base_width)``; dimensionless shape
            metric (∼0.5 for triangular, ∼0.79 for Gaussian, →1 rectangular).
        ``asymmetry_hw``
            ``(wl_right_half − wl_min) / (wl_min − wl_left_half)``; ratio of
            right to left half-widths at half maximum.  ``nan`` when the
            left half-width is zero.
        ``asymmetry_centroid``
            ``(wl_center − wl_min) / fwhm``; signed centroid offset normalized
            by FWHM.  Positive values indicate absorption weight skewed toward
            longer wavelengths.
    """
    lo, hi = wl_range
    mask   = (xaxis >= lo) & (xaxis <= hi)
    if mask.sum() < 3:
        return None

    x_sub  = xaxis[mask]
    cr     = remove_continuum(xaxis, spectrum, wl_range=wl_range)
    cr_sub = cr[mask]

    i_min      = int(np.argmin(cr_sub))
    band_depth = float(1.0 - cr_sub[i_min])
    if band_depth < 1e-4:
        return None   # no detectable absorption

    wl_min = float(x_sub[i_min])

    # ── Spectral centroid ────────────────────────────────────────────────────
    absorption = 1.0 - cr_sub
    wl_center  = float(np.dot(x_sub, absorption) / absorption.sum())

    # ── Interpolated crossing helper ─────────────────────────────────────────
    def _left_crossing(level: float) -> float:
        above = np.where(cr_sub[:i_min] >= level)[0]
        if len(above) == 0:
            return float(x_sub[0])
        j  = int(above[-1])
        dy = cr_sub[j + 1] - cr_sub[j]
        t  = (level - cr_sub[j]) / dy if abs(dy) > 1e-12 else 0.0
        return float(x_sub[j] + t * (x_sub[j + 1] - x_sub[j]))

    def _right_crossing(level: float) -> float:
        above = np.where(cr_sub[i_min + 1:] >= level)[0]
        if len(above) == 0:
            return float(x_sub[-1])
        j  = int(above[0]) + i_min + 1
        dy = cr_sub[j] - cr_sub[j - 1]
        t  = (level - cr_sub[j - 1]) / dy if abs(dy) > 1e-12 else 0.0
        return float(x_sub[j - 1] + t * (x_sub[j] - x_sub[j - 1]))

    # ── FWHM crossings (50 % depth) ──────────────────────────────────────────
    half      = 1.0 - band_depth / 2.0
    wl_left   = _left_crossing(half)
    wl_right  = _right_crossing(half)
    fwhm      = max(wl_right - wl_left, 0.0)

    # ── Base-width crossings (10 % depth) ────────────────────────────────────
    base_level      = 1.0 - 0.10 * band_depth
    wl_left_base    = _left_crossing(base_level)
    wl_right_base   = _right_crossing(base_level)
    base_width      = max(wl_right_base - wl_left_base, 0.0)

    # ── Band area over 10 %-threshold extent ─────────────────────────────────
    base_mask = (x_sub >= wl_left_base) & (x_sub <= wl_right_base)
    x_base    = x_sub[base_mask]
    cr_base   = cr_sub[base_mask]
    band_area = float(simpson(1.0 - cr_base, x=x_base)) if len(x_base) >= 2 else 0.0

    band_area_ratio = band_area / (band_depth * base_width) if base_width > 0 else 0.0

    # ── Asymmetry: half-width ratio ──────────────────────────────────────────
    left_hw      = wl_min - wl_left
    right_hw     = wl_right - wl_min
    asymmetry_hw = right_hw / left_hw if left_hw > 1e-6 else float('nan')

    # ── Asymmetry: centroid offset normalized by FWHM ────────────────────────
    asymmetry_centroid = (wl_center - wl_min) / fwhm if fwhm > 0 else 0.0

    return {
        'wl_center':          wl_center,
        'wl_min':             wl_min,
        'band_depth':         band_depth,
        'fwhm':               fwhm,
        'base_width':         base_width,
        'band_area':          band_area,
        'band_area_ratio':    band_area_ratio,
        'asymmetry_hw':       asymmetry_hw,
        'asymmetry_centroid': asymmetry_centroid,
    }


# =============================================================================
# ========================= moving_average =====================================
# =============================================================================

def moving_average(data: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """
    Simple centered moving-average (boxcar) smoothing along *axis*.

    A window-based smoother expressed directly in samples (channels), suitable
    for any spectral axis regardless of units.  Edges use ``'nearest'`` handling
    so the output length matches the input.

    Parameters
    ----------
    data : np.ndarray
        Input array; smoothing is applied along *axis*.  For a stack of spectra
        of shape ``(n_spectra, n_bands)`` the default *axis* smooths each
        spectrum independently.
    window : int
        Boxcar width in samples.  Values ``<= 1`` return an unmodified copy.
    axis : int
        Axis along which to smooth.  Default ``-1`` (the spectral axis).

    Returns
    -------
    np.ndarray
        Smoothed array, same shape as *data*, dtype float64.
    """
    arr = np.asarray(data, dtype=float)
    if window is None or window <= 1:
        return arr.copy()
    return uniform_filter1d(arr, size=int(window), axis=axis, mode='nearest')


# =============================================================================
# ========================= smooth_spectrum ====================================
# =============================================================================

def smooth_spectrum(
    xaxis:    np.ndarray,
    spectrum: np.ndarray,
    method:   str   = 'savgol',
    *,
    window_nm:  float = 30.0,
    poly_order: int   = 3,
) -> np.ndarray:
    """
    Apply spectral smoothing to a single reflectance spectrum.

    The smoothing scale is always expressed in nanometres and converted to
    samples using the median channel spacing of *xaxis*, making the result
    portable across instruments with different spectral sampling.

    Parameters
    ----------
    xaxis : np.ndarray
        Wavelength axis, shape (n_channels,).
    spectrum : np.ndarray
        Single spectrum, shape (n_channels,).
    method : str
        Smoothing algorithm.  One of:

        ``'moving_avg'``
            Boxcar (uniform) convolution.
        ``'moving_median'``
            Running median.
        ``'savgol'``
            Savitzky-Golay filter (default).  *window_nm* is converted to the
            nearest odd sample count that is also > *poly_order*.
        ``'gaussian'``
            Gaussian kernel.  *window_nm* sets the full-width at half-maximum
            (σ = window_nm / 2.355).
    window_nm : float
        Smoothing scale in nm.  For Savitzky-Golay the converted window is
        rounded to the nearest odd sample count ≥ ``poly_order + 1``; the
        tooltip in the GUI notes this rounding.
    poly_order : int
        Polynomial order for ``'savgol'``.  Ignored by other methods.

    Returns
    -------
    np.ndarray
        Smoothed spectrum, same shape as *spectrum*, dtype float64.

    Raises
    ------
    ValueError
        If *method* is not one of the four recognised strings.
    """
    nm_per_sample = float(np.median(np.diff(xaxis)))
    n_samples     = max(1, round(window_nm / nm_per_sample))
    arr           = spectrum.astype(float)

    if method == 'moving_avg':
        return uniform_filter1d(arr, size=n_samples, mode='nearest')

    if method == 'moving_median':
        return median_filter(arr, size=n_samples, mode='nearest')

    if method == 'savgol':
        win     = n_samples if n_samples % 2 == 1 else n_samples + 1
        min_win = poly_order + 1 if (poly_order + 1) % 2 == 1 else poly_order + 2
        win     = max(win, min_win)
        return savgol_filter(arr, window_length=win, polyorder=poly_order,
                             mode='nearest')

    if method == 'gaussian':
        sigma = (window_nm / 2.355) / nm_per_sample   # FWHM → σ in samples
        return gaussian_filter1d(arr, sigma=sigma, mode='nearest')

    raise ValueError(
        f"Unknown smoothing method '{method}'. "
        "Choose from 'moving_avg', 'moving_median', 'savgol', 'gaussian'."
    )


# =============================================================================
# =========================== detect_bands ====================================
# =============================================================================

def detect_bands(
    xaxis:    np.ndarray,
    spectrum: np.ndarray,
    presets:  list[dict] | None = None,
    *,
    smooth_method:      str   = 'savgol',
    smooth_window_nm:   float = 30.0,
    smooth_polyorder:   int   = 3,
    min_prominence:     float = 0.02,
    min_width_nm:       float = 15.0,
    min_depth:          float = 0.01,
    match_tolerance_nm: float = 30.0,
) -> list[dict]:
    """
    Automatically detect absorption features in a reflectance spectrum.

    The spectrum is smoothed, local minima are identified via prominence- and
    width-filtered peak detection, each candidate is characterised with
    :func:`band_parameters` on the *raw* (unsmoothed) spectrum using an
    auto-derived shoulder window, and results are optionally matched against
    a list of known preset features.

    Parameters
    ----------
    xaxis : np.ndarray
        Wavelength axis, shape (n_channels,).  Must be monotonically increasing.
    spectrum : np.ndarray
        Single reflectance spectrum, shape (n_channels,).  Values in [0, 1].
    presets : list[dict] or None
        Preset feature list (keys: ``name``, ``wavelength``, ``fwhm``,
        ``wl_range``).  Loaded from the YAML by the viewer.  ``None`` skips
        matching.
    smooth_method : str
        Passed to :func:`smooth_spectrum`.
    smooth_window_nm : float
        Smoothing scale in nm, passed to :func:`smooth_spectrum`.
    smooth_polyorder : int
        Savitzky-Golay polynomial order, passed to :func:`smooth_spectrum`.
    min_prominence : float
        Minimum peak prominence in reflectance units [0–1] for
        ``scipy.signal.find_peaks``.  Controls sensitivity to shallow features
        relative to their local background.
    min_width_nm : float
        Minimum peak width in nm, converted to samples for ``find_peaks``.
    min_depth : float
        Minimum ``band_depth`` (from :func:`band_parameters`, after continuum
        removal) to retain a candidate.
    match_tolerance_nm : float
        A detected ``wl_min`` is matched to a preset when the distance to its
        nominal ``wavelength`` is within this tolerance.  The closest preset
        within tolerance is selected.

    Returns
    -------
    list[dict]
        Candidates sorted by ``wl_min``.  Each entry contains all keys from
        :func:`band_parameters` plus:

        ``wl_range``
            ``(wl_lo, wl_hi)`` shoulder window used for continuum removal.
        ``matched_name``
            Name of the closest preset within tolerance, or ``None``.
        ``matched_preset``
            The full preset dict, or ``None``.
    """
    smoothed = smooth_spectrum(
        xaxis, spectrum, smooth_method,
        window_nm=smooth_window_nm, poly_order=smooth_polyorder,
    )

    nm_per_sample     = float(np.median(np.diff(xaxis)))
    min_width_samples = max(1, round(min_width_nm / nm_per_sample))

    # ── Detect minima in smoothed spectrum ───────────────────────────────────
    min_indices, _ = find_peaks(
        -smoothed,
        prominence=min_prominence,
        width=min_width_samples,
    )
    if len(min_indices) == 0:
        return []

    # ── Shoulder candidates: local maxima in smoothed + endpoints ────────────
    max_indices, _ = find_peaks(smoothed)
    shoulders = np.concatenate([[0], max_indices, [len(xaxis) - 1]])

    candidates: list[dict] = []
    for i_peak in min_indices:
        left_sh  = shoulders[shoulders < i_peak]
        right_sh = shoulders[shoulders > i_peak]
        i_left   = int(left_sh[-1])  if len(left_sh)  > 0 else 0
        i_right  = int(right_sh[0])  if len(right_sh) > 0 else len(xaxis) - 1

        wl_range = (float(xaxis[i_left]), float(xaxis[i_right]))

        bp = band_parameters(xaxis, spectrum, wl_range)
        if bp is None or bp['band_depth'] < min_depth:
            continue

        # ── Preset matching: closest within tolerance ─────────────────────
        matched_name:   str  | None = None
        matched_preset: dict | None = None
        if presets:
            best_dist = match_tolerance_nm
            for preset in presets:
                dist = abs(bp['wl_min'] - float(preset['wavelength']))
                if dist < best_dist:
                    best_dist      = dist
                    matched_name   = preset['name']
                    matched_preset = preset

        candidates.append({
            **bp,
            'wl_range':       wl_range,
            'matched_name':   matched_name,
            'matched_preset': matched_preset,
        })

    candidates.sort(key=lambda c: c['wl_min'])
    return candidates


# =============================================================================
# ==================== batch VSWIR processing =================================
# =============================================================================

def smooth_spectra(
    data: dict,
    method: str = 'savgol',
    *,
    window_nm:  float = 30.0,
    poly_order: int   = 3,
) -> dict:
    """
    Apply spectral smoothing to every spectrum in a data dict.

    A thin batch wrapper around :func:`smooth_spectrum`.  All spectra in
    ``data['spectra']`` must share the common ``data['xaxis']`` axis (which is
    the contract established by :func:`~speclab.utils.load_reflectance_vswir`).

    Parameters
    ----------
    data : dict
        Dict with at minimum keys ``'xaxis'`` (shape ``(n_channels,)``) and
        ``'spectra'`` (``dict[str, np.ndarray]``, each 1-D, shape
        ``(n_channels,)``).  Any extra keys are forwarded to the output.
    method : str
        Smoothing algorithm passed to :func:`smooth_spectrum`.
        One of ``'savgol'`` (default), ``'moving_avg'``, ``'moving_median'``,
        ``'gaussian'``.
    window_nm : float
        Smoothing scale in nanometres (default 30).
    poly_order : int
        Savitzky-Golay polynomial order (default 3, ignored by other methods).

    Returns
    -------
    dict
        Copy of *data* with ``'spectra'`` replaced by smoothed arrays and
        ``'smooth_params'`` added::

            {
                'xaxis':        np.ndarray,
                'spectra':      dict[str, np.ndarray],   # smoothed
                'smooth_params': {'method': str, 'window_nm': float,
                                  'poly_order': int},
                ...                                       # forwarded keys
            }
    """
    xaxis = data['xaxis']
    smoothed_spectra: dict[str, np.ndarray] = {
        name: smooth_spectrum(xaxis, arr, method,
                              window_nm=window_nm, poly_order=poly_order)
        for name, arr in data['spectra'].items()
    }
    return {
        **data,
        'spectra':       smoothed_spectra,
        'smooth_params': {
            'method':     method,
            'window_nm':  window_nm,
            'poly_order': poly_order,
        },
    }


def remove_continuum_batch(
    data:     dict,
    wl_range: tuple[float, float] | None = None,
) -> dict:
    """
    Apply convex-hull continuum removal to every spectrum in a data dict.

    A thin batch wrapper around :func:`remove_continuum`.  Returns continuum-
    removed (CR) reflectance where 1.0 means "on the continuum" and values
    below 1.0 indicate absorption depth relative to the local baseline.

    Parameters
    ----------
    data : dict
        Dict with ``'xaxis'`` and ``'spectra'`` (see :func:`smooth_spectra`).
    wl_range : tuple[float, float] or None
        ``(wl_lo, wl_hi)`` shoulder window for the hull computation (nm).
        Channels outside this range are set to 1.0.  ``None`` uses the full
        spectrum.

    Returns
    -------
    dict
        Copy of *data* with ``'spectra'`` replaced by CR arrays and
        ``'cr_params'`` added::

            {
                'xaxis':     np.ndarray,
                'spectra':   dict[str, np.ndarray],  # continuum-removed
                'cr_params': {'wl_range': tuple or None},
                ...
            }
    """
    xaxis = data['xaxis']
    cr_spectra: dict[str, np.ndarray] = {
        name: remove_continuum(xaxis, arr, wl_range=wl_range)
        for name, arr in data['spectra'].items()
    }
    return {
        **data,
        'spectra':   cr_spectra,
        'cr_params': {'wl_range': wl_range},
    }


def band_parameters_batch(
    data:     dict,
    features: list[dict],
) -> dict:
    """
    Compute band parameters for a list of features on every spectrum in a data dict.

    Runs :func:`band_parameters` for each (spectrum, feature) pair.  Features
    absent from a spectrum (too few channels, no detectable absorption) produce
    a ``None`` entry rather than raising.

    Parameters
    ----------
    data : dict
        Dict with ``'xaxis'`` and ``'spectra'`` (see :func:`smooth_spectra`).
        Pass continuum-removed data (from :func:`remove_continuum_batch`) for
        more precise parameter estimates, or smoothed data (from
        :func:`smooth_spectra`) for noise reduction.
    features : list[dict]
        Feature descriptors.  Each dict must have at minimum:

        ``'name'``
            Unique string identifier (used as the key in the results).
        ``'wl_range'``
            ``(wl_lo, wl_hi)`` shoulder window passed to
            :func:`band_parameters`.

        Additional keys (``'group'``, ``'wavelength'``, ``'fwhm'``, …) are
        carried through unchanged into ``results['features']``.

    Returns
    -------
    dict
        Keys::

            {
                'xaxis':    np.ndarray,
                'features': list[dict],               # input feature list
                'results':  dict[str, dict[str, dict | None]],
            }

        ``results[spectrum_name][feature_name]`` is either the dict returned
        by :func:`band_parameters` (keys: ``wl_center``, ``wl_min``,
        ``band_depth``, ``fwhm``, ``base_width``, ``band_area``,
        ``band_area_ratio``, ``asymmetry_hw``, ``asymmetry_centroid``) or
        ``None`` when the feature is absent.
    """
    xaxis = data['xaxis']
    results: dict[str, dict[str, dict | None]] = {}
    for name, arr in data['spectra'].items():
        feat_results: dict[str, dict | None] = {}
        for feat in features:
            feat_results[feat['name']] = band_parameters(
                xaxis, arr, wl_range=feat['wl_range'])
        results[name] = feat_results
    return {
        'xaxis':    xaxis,
        'features': features,
        'results':  results,
    }


def detect_bands_batch(
    data:    dict,
    presets: list[dict] | None = None,
    *,
    wl_range:           tuple[float, float] | None = None,
    smooth_method:      str   = 'savgol',
    smooth_window_nm:   float = 30.0,
    smooth_polyorder:   int   = 3,
    min_prominence:     float = 0.02,
    min_width_nm:       float = 15.0,
    min_depth:          float = 0.01,
    match_tolerance_nm: float = 30.0,
) -> dict:
    """
    Detect absorption features in every spectrum of a data dict and merge
    candidates across spectra.

    Runs :func:`detect_bands` on each spectrum, then merges per-spectrum
    candidate lists into a single cross-spectrum list: the first occurrence's
    parameters are kept and successive matching detections only append the
    spectrum name to ``'seen_in'``.

    Parameters
    ----------
    data : dict
        Dict with ``'xaxis'`` and ``'spectra'`` (see :func:`smooth_spectra`).
    presets : list[dict] or None
        Known feature descriptors used for name-matching (keys: ``'name'``,
        ``'wavelength'``, ``'fwhm'``, ``'wl_range'``).  ``None`` skips
        matching.
    wl_range : tuple[float, float] or None
        Restrict detection to this wavelength window (nm).  Channels outside
        are excluded before passing to :func:`detect_bands`.  Useful for
        suppressing MIR bands when analysing VSWIR data that extends beyond
        2500 nm.
    smooth_method : str
        Smoothing algorithm (see :func:`smooth_spectrum`).
    smooth_window_nm : float
        Smoothing window in nm (default 30).
    smooth_polyorder : int
        Savitzky-Golay polynomial order (default 3).
    min_prominence : float
        Minimum peak prominence in reflectance units [0–1].
    min_width_nm : float
        Minimum peak width in nm.
    min_depth : float
        Minimum band depth after continuum removal to retain a candidate.
    match_tolerance_nm : float
        Two candidates from different spectra are merged when their ``wl_min``
        values differ by less than this value (nm).  Also used as the preset-
        matching tolerance inside :func:`detect_bands`.

    Returns
    -------
    dict
        Keys::

            {
                'per_spectrum': dict[str, list[dict]],  # raw per-spectrum results
                'merged':       list[dict],              # cross-spectrum merged list
                'params':       dict,                    # all detection settings
            }

        Each entry in ``'merged'`` contains all keys from
        :func:`band_parameters` plus ``'wl_range'``, ``'matched_name'``,
        ``'matched_preset'``, and ``'seen_in'`` (list of spectrum names).
    """
    xaxis = data['xaxis']
    if wl_range is not None:
        mask  = (xaxis >= wl_range[0]) & (xaxis <= wl_range[1])
        x_sub = xaxis[mask]
    else:
        mask  = np.ones(len(xaxis), dtype=bool)
        x_sub = xaxis

    per_spectrum: dict[str, list[dict]] = {}
    merged: list[dict] = []
    for name, arr in data['spectra'].items():
        candidates = detect_bands(
            x_sub, arr[mask], presets,
            smooth_method=smooth_method,
            smooth_window_nm=smooth_window_nm,
            smooth_polyorder=smooth_polyorder,
            min_prominence=min_prominence,
            min_width_nm=min_width_nm,
            min_depth=min_depth,
            match_tolerance_nm=match_tolerance_nm,
        )
        per_spectrum[name] = candidates
        for cand in candidates:
            existing = next(
                (m for m in merged
                 if abs(cand['wl_min'] - m['wl_min']) < match_tolerance_nm),
                None,
            )
            if existing is not None:
                existing['seen_in'].append(name)
            else:
                merged.append({**cand, 'seen_in': [name]})

    merged.sort(key=lambda c: c['wl_min'])
    return {
        'per_spectrum': per_spectrum,
        'merged':       merged,
        'params': {
            'wl_range':           wl_range,
            'smooth_method':      smooth_method,
            'smooth_window_nm':   smooth_window_nm,
            'smooth_polyorder':   smooth_polyorder,
            'min_prominence':     min_prominence,
            'min_width_nm':       min_width_nm,
            'min_depth':          min_depth,
            'match_tolerance_nm': match_tolerance_nm,
        },
    }


# =============================================================================
# ============================== load_sbm =====================================
# =============================================================================

_BB_WARM_PATTERNS: list[str] = ["bbcold", "bbc", "coldbb", "bbwarm", "bbw", "warmbb"]
_BB_HOT_PATTERNS:  list[str] = ["bbhot", "bbh", "hotbb"]


class MissingTempsError(Exception):
    """Raised when BB or downwelling temperatures cannot be determined."""
    pass


def _parse_bb_input(raw: str) -> float:
    """
    Parse a blackbody temperature entry and return the temperature in K.

    Accepts either a single value in °C or a whitespace/comma-separated
    resistance pair ``ch1 ch2`` (converted via :func:`utils.r2t_nau`).

    Parameters
    ----------
    raw : str
        Raw user input string.

    Returns
    -------
    float
        Temperature in K.

    Raises
    ------
    ValueError
        If the input cannot be parsed as one or two numbers.
    """
    parts = raw.replace(',', ' ').split()
    if len(parts) == 1:
        return utils.c2k(float(parts[0]))
    elif len(parts) == 2:
        return utils.r2t_nau(float(parts[0]), float(parts[1]))
    raise ValueError(f"Expected 1 value (°C) or 2 values (ch1 ch2), got: {raw!r}")


def _prompt_bb_temp(name: str) -> float:
    """
    Prompt for a single blackbody temperature via stdin and return it in K.

    Accepts either a temperature in °C or a ``ch1 ch2`` resistance pair.

    Parameters
    ----------
    name : str
        Human-readable BB name shown in the prompt (e.g. ``"warm BB"``).

    Returns
    -------
    float
        Temperature in K.
    """
    print(f"No measurement info file found — enter {name} temperature.")
    print("  Enter a temperature in °C, or a resistance pair as 'ch1 ch2'.")
    while True:
        try:
            return _parse_bb_input(input(f"  {name}: "))
        except ValueError as exc:
            print(f"  Invalid input ({exc}) — try again.")


def _prompt_downwelling_temps(labels: list[str]) -> dict[str, float]:
    """
    Collect per-sample downwelling temperatures interactively from stdin.

    Prints a prompt for each label and reads a float (°C) from the user.
    Re-prompts on invalid input.  Used by :func:`emcal` when ``lab='nau'``
    and no measurement info file is found and no ``downwelling_temps`` dict
    was supplied.

    Parameters
    ----------
    labels : list[str]
        Ordered list of sample labels.

    Returns
    -------
    dict[str, float]
        Mapping of sample label → downwelling temperature in °C.
    """
    print("No measurement info file found.")
    print("Enter the downwelling (ambient) temperature in °C for each sample.")
    temps: dict[str, float] = {}
    for label in labels:
        while True:
            try:
                val = float(input(f"  {label}: "))
                temps[label] = val
                break
            except ValueError:
                print("  Invalid value — please enter a number.")
    return temps


def _load_notes(
    fdir_or_path: str,
) -> tuple['pd.DataFrame | None', list[str]]:
    """
    Load measurement notes, preferring CSV/XLS/XLSX over a legacy TXT file.

    Tries :func:`utils.readEmissionCSVnotes` first; on failure tries
    :func:`utils.readEmissionTXTnotes`.  Returns ``(None, [])`` when neither
    is found.

    Parameters
    ----------
    fdir_or_path : str
        Folder to search, or explicit path to a notes file.

    Returns
    -------
    tuple[pd.DataFrame | None, list[str]]
        ``(df, flist)`` — the parsed DataFrame and list of file paths
        involved, or ``(None, [])`` if no notes file is found.
    """
    try:
        df, flist = utils.readEmissionCSVnotes(fdir_or_path, return_path=True)
        return df, flist
    except IOError:
        pass
    try:
        df, flist = utils.readEmissionTXTnotes(fdir_or_path, save=False, return_path=True)
        return df, flist
    except IOError:
        pass
    return None, []


def scan_sample_labels(fdir: str, ext: str = ".csv") -> list[str]:
    """
    Return sample labels from a measurement folder without loading spectra.

    Applies the same file-exclusion rules as :func:`cal_rad`: blackbody
    files, previous results, and the measurement notes file (if present)
    are all excluded.  Only filenames are read; no spectra are parsed.

    Parameters
    ----------
    fdir : str
        Path to the measurement folder.  ``~`` is expanded.
    ext : str
        File extension to search for (e.g. ``".csv"``).

    Returns
    -------
    list[str]
        Sample labels in sorted order, derived from filenames without
        their extension.
    """
    if "~" in fdir:
        fdir = os.path.expanduser(fdir)
    fdir = os.path.abspath(fdir)

    bbc_flist = utils.findFiles(_BB_WARM_PATTERNS, ext, fdir)
    bbh_flist = utils.findFiles(_BB_HOT_PATTERNS,  ext, fdir)
    previous  = utils.findFiles(['emcal', 'results'], ext, fdir)
    _, note_flist = _load_notes(fdir)

    flist = utils.findFiles("", ext, fdir)
    flist = [f for f in flist
             if f not in bbc_flist + bbh_flist + note_flist + previous]
    flist.sort()
    return [os.path.splitext(os.path.basename(f))[0] for f in flist]


def load_sbm(
    fdir: str,
    ext: str = ".csv",
    lab: str = "nau",
    notes_path: str | None = None,
) -> dict:
    """
    Load raw single-beam spectra from a measurement folder.

    Identifies and excludes blackbody files, notes files, and previous
    calibration results, then loads all remaining spectra.  Called
    internally by :func:`emcal` and usable standalone for SBM display
    before running calibration.

    Parameters
    ----------
    fdir : str
        Path to the folder.  ``~`` is expanded.
    ext : str
        File extension to search for (e.g. ``".csv"``).
    lab : str
        Lab convention used to locate the notes file when ``lab='nau'``.
    notes_path : str or None
        Explicit path to the notes CSV, or ``None`` to search inside *fdir*.

    Returns
    -------
    dict
        ``xaxis``  : np.ndarray — full wavenumber axis (cm⁻¹), before any SBM mask
        ``sbm``    : dict[str, np.ndarray] — {label: single-beam array}
        ``label``  : list[str] — sample labels in load order
        ``fdir``   : str — absolute folder path
    """
    if "~" in fdir:
        fdir = os.path.expanduser(fdir)
    fdir = os.path.abspath(fdir)

    if lab == "nau":
        _, note_flist = _load_notes(notes_path if notes_path is not None else fdir)
    else:
        note_flist = []

    previous_results = utils.findFiles(['emcal', 'results'], ext, fdir)
    bbc_flist        = utils.findFiles(_BB_WARM_PATTERNS,    ext, fdir)
    bbh_flist        = utils.findFiles(_BB_HOT_PATTERNS,     ext, fdir)
    log_flist        = utils.findFiles(['live-log'],          ext, fdir)

    flist = utils.findFiles("", ext, fdir)
    flist = [f for f in flist
             if f not in bbc_flist + bbh_flist + note_flist + previous_results + log_flist]
    flist.sort()

    if not flist:
        raise IOError(f"No sample spectra found in {fdir}")

    labels: list[str] = []
    sbm: dict[str, np.ndarray] = {}
    xaxis: np.ndarray | None = None
    skipped: list[str] = []

    for fname in flist:
        label = os.path.splitext(os.path.basename(fname))[0]
        # Skip and warn on files that are not readable two-column spectra
        # (e.g. cloud-sync "conflicted copy" notes files that slip past the
        # exclusion filters) so one bad file does not abort the whole folder.
        try:
            spec = utils.readOMNIC(fname)
        except Exception as exc:
            skipped.append(label)
            logging.warning("load_sbm: skipping unreadable file '%s' (%s)",
                            os.path.basename(fname), exc)
            continue
        if xaxis is None:
            xaxis = spec['wn']
        elif len(spec['wn']) != len(xaxis) or not np.all(spec['wn'] == xaxis):
            skipped.append(label)
            logging.warning("load_sbm: skipping '%s' (wavenumber axis mismatch)", label)
            continue
        labels.append(label)
        sbm[label] = spec['data']

    if not labels:
        raise IOError(f"No readable sample spectra found in {fdir}")
    if skipped:
        logging.warning("load_sbm: skipped %d file(s) in %s: %s",
                        len(skipped), fdir, ', '.join(skipped))

    return {'xaxis': xaxis, 'sbm': sbm, 'label': labels, 'fdir': fdir}


def _fit_planck_to_ri(
    ri: np.ndarray,
    wn: np.ndarray,
    t_min: float = 250.0,
    t_max: float = 400.0,
    n_temps: int = 50,
) -> np.ndarray:
    """
    Fit a library of Planck functions to instrument radiance *ri*.

    Mirrors spectral_tools.dvrc::irf_ri ``noise_free=1``, which calls
    ``fit_bb(ri, xaxis, type='inst', wave1=300, wave2=2500)``.  A set of
    Planck spectra at *n_temps* temperatures spanning [*t_min*, *t_max*]
    is fitted via NNLS; the modelled spectrum is returned.

    Parameters
    ----------
    ri : np.ndarray
        Raw instrument radiance (1-D, length n_bands).
    wn : np.ndarray
        Masked wavenumber axis (cm⁻¹), same length as *ri*.
    t_min, t_max : float
        Temperature bounds (K) for the Planck library.
    n_temps : int
        Number of Planck spectra in the library.

    Returns
    -------
    np.ndarray
        Noise-free instrument radiance, same shape as *ri*.
    """
    temps = np.linspace(t_min, t_max, n_temps)
    lib   = np.column_stack([utils.rad(wn, t) for t in temps])   # (n_bands, n_temps)
    coeffs, _ = nnls(lib, ri)
    return lib @ coeffs


# =============================================================================
# ============================== cal_rad ======================================
# =============================================================================

def cal_rad(
    fdir: str,
    bb1: float | tuple[float, float] | None = None,
    bb2: float | tuple[float, float] | None = None,
    ext: str = ".csv",
    bb_emiss: float = 0.995,
    sbm_threshold: float = 0.05,
    lab: str = "nau",
    notes_path: str | None = None,
    fir: bool = False,
    exclude: tuple[float, float] | None = None,
    noise_free: bool = True,
    on_missing_bb_temps: Callable[[], tuple[float, float]] | None = None,
) -> dict:
    """
    Perform radiance calibration on a folder of FTIR spectra.

    Loads two blackbody spectra, derives the instrument response function
    (IRF), and converts sample single-beam spectra to calibrated radiance.
    This is the first stage of :func:`emcal`; call :func:`emcal` to also
    retrieve emissivity.

    Ref: spectral_tools.dvrc::irf_ri + cal_rad.

    Parameters
    ----------
    fdir : str
        Path to the folder containing spectra.  ``~`` is expanded.
    bb1 : float or tuple[float, float] or None
        Warm blackbody temperature (K), or a (ch1_res, ch2_res) resistance
        pair.  If None and ``lab='nau'``, read from the notes file.
    bb2 : float or tuple[float, float] or None
        Hot blackbody temperature (K), or resistance pair.
    ext : str
        File extension to search for (e.g. ``".csv"``).
    bb_emiss : float
        Assumed emissivity of the blackbody cavities.
    sbm_threshold : float
        Minimum (hot − warm) SBM difference to retain a wavenumber channel.
    lab : str
        Lab convention for BB temperature conversion: ``"nau"``, ``"asu"``,
        or ``"swri"``.
    notes_path : str or None
        Explicit path to the measurement notes CSV.  If None, searched
        automatically inside *fdir*.  Only used when ``lab='nau'`` and BB
        temperatures are not supplied directly.
    fir : bool
        Force far-IR mode (overrides automatic detection).
    exclude : tuple[float, float] or None
        Wavenumber interval ``(wn_low, wn_high)`` to exclude.
    noise_free : bool
        If True (default), fit a Planck library to the raw instrument
        radiance to obtain a noise-free ``ri``, then average two IRF
        estimates (one per BB) to reduce IRF noise by √2.
        Ref: spectral_tools.dvrc::irf_ri, ``noise_free=1``.
    on_missing_bb_temps : callable or None
        Called when ``lab='nau'``, the measurement info file cannot be
        found, and at least one of *bb1* / *bb2* was not supplied.
        Must return ``(bb1_K, bb2_K)`` — both temperatures in K.
        If None, falls back to interactive terminal prompts.
        Raise :exc:`MissingTempsError` inside the callback to abort.

    Returns
    -------
    dict
        ``xaxis``   : np.ndarray — masked wavenumber axis (cm⁻¹)
        ``wl``      : np.ndarray — wavelength axis (µm)
        ``calib``   : dict — IRF, ri, BB temperatures and radiances
        ``sbm``     : dict — masked single-beam per label + ``bbc``/``bbh``
        ``rad``     : dict — calibrated radiance per label + ``bbc``/``bbh``
        ``label``   : list[str] — ordered sample labels
        ``_fir``    : bool — resolved far-IR flag (internal)
    """
    if "~" in fdir:
        fdir = os.path.expanduser(fdir)
    fdir = os.path.abspath(fdir)

    if lab == "nau":
        notes, _ = _load_notes(notes_path if notes_path is not None else fdir)
        _notes_loaded = notes is not None
        if not _notes_loaded:
            logging.warning(
                "Measurement info file not found — BB temperatures unavailable from notes: %s",
                fdir,
            )

    # Resolve BB temperatures
    # Call on_missing_bb_temps once when (a) the notes file is absent, or
    # (b) the notes are present but BB resistance columns are NaN/empty (e.g.
    # the multimeter was not connected during BB collection).  The callback
    # must return (bb1_K, bb2_K); raise MissingTempsError inside it to abort.
    _bbc_pat = '|'.join(_BB_WARM_PATTERNS)
    _bbh_pat = '|'.join(_BB_HOT_PATTERNS)
    _cb_bb1_k: float | None = None
    _cb_bb2_k: float | None = None

    _need_bb_callback = False
    if lab == "nau" and (bb1 is None or bb2 is None):
        if not _notes_loaded:
            _need_bb_callback = True
        else:
            # Notes loaded — check whether resistance columns are usable.
            if bb1 is None:
                _bbc_q = notes[notes["sample_name"].str.contains(
                    _bbc_pat, case=False, regex=True)]
                if not _bbc_q.empty:
                    _r1 = _bbc_q["channel_101"].values[0]
                    _r2 = _bbc_q["channel_102"].values[0]
                    if pd.isna(_r1) or pd.isna(_r2):
                        _need_bb_callback = True
            if not _need_bb_callback and bb2 is None:
                _bbh_q = notes[notes["sample_name"].str.contains(
                    _bbh_pat, case=False, regex=True)]
                if not _bbh_q.empty:
                    _r1 = _bbh_q["channel_101"].values[0]
                    _r2 = _bbh_q["channel_102"].values[0]
                    if pd.isna(_r1) or pd.isna(_r2):
                        _need_bb_callback = True
            if _need_bb_callback:
                logging.warning(
                    "BB resistance values are NaN in measurement info "
                    "(multimeter not connected during BB collection?). "
                    "Requesting temperatures via on_missing_bb_temps.")

    if _need_bb_callback:
        if on_missing_bb_temps is not None:
            _cb_bb1_k, _cb_bb2_k = on_missing_bb_temps()
        else:
            if bb1 is None:
                _cb_bb1_k = _prompt_bb_temp("warm BB")
            if bb2 is None:
                _cb_bb2_k = _prompt_bb_temp("hot BB")

    if bb1 is None and lab == "nau":
        # Use callback result when notes were absent or had NaN resistances.
        if _notes_loaded and _cb_bb1_k is None:
            bbc_row = notes[notes["sample_name"].str.contains(_bbc_pat, case=False, regex=True)]
            bbc_res_ch1 = bbc_row["channel_101"].values[0]
            bbc_res_ch2 = bbc_row["channel_102"].values[0]
            bbc_temp = utils.r2t_nau(bbc_res_ch1, bbc_res_ch2)
        else:
            bbc_temp = _cb_bb1_k
    if isinstance(bb1, (int, float)):
        bbc_temp = bb1
    elif isinstance(bb1, tuple):
        bbc_res_ch1, bbc_res_ch2 = bb1
        if lab == "asu":
            bbc_temp = utils.r2t_lo(bbc_res_ch1, bbc_res_ch2)
        elif lab == "swri":
            bbc_temp = utils.r2t_swri(bbc_res_ch1, bbc_res_ch2)
        elif lab == "nau":
            bbc_temp = utils.r2t_nau(bbc_res_ch1, bbc_res_ch2)

    if bb2 is None and lab == "nau":
        # Use callback result when notes were absent or had NaN resistances.
        if _notes_loaded and _cb_bb2_k is None:
            bbh_row = notes[notes["sample_name"].str.contains(_bbh_pat, case=False, regex=True)]
            bbh_res_ch1 = bbh_row["channel_101"].values[0]
            bbh_res_ch2 = bbh_row["channel_102"].values[0]
            bbh_temp = utils.r2t_nau(bbh_res_ch1, bbh_res_ch2)
        else:
            bbh_temp = _cb_bb2_k
    if isinstance(bb2, (int, float)):
        bbh_temp = bb2
    elif isinstance(bb2, tuple):
        bbh_res_ch1, bbh_res_ch2 = bb2
        if lab == "asu":
            bbh_temp = utils.r2t_lo(bbh_res_ch1, bbh_res_ch2)
        elif lab == "swri":
            bbh_temp = utils.r2t_swri(bbh_res_ch1, bbh_res_ch2)
        elif lab == "nau":
            bbh_temp = utils.r2t_nau(bbh_res_ch1, bbh_res_ch2)

    # Load BBs
    bbc_flist = utils.findFiles(_BB_WARM_PATTERNS, ext, fdir)
    if len(bbc_flist) == 0:
        raise IOError("Could not find warm BB file in %s" % fdir)
    elif len(bbc_flist) > 1:
        raise IOError("Found multiple potential files for warm BB: %s" % bbc_flist)
    bbc = utils.readOMNIC(bbc_flist[0])
    logging.info("Found blackbody spectrum: %s", bbc_flist[0])

    bbh_flist = utils.findFiles(_BB_HOT_PATTERNS, ext, fdir)
    if len(bbh_flist) == 0:
        raise IOError("Could not find hot BB file in %s" % fdir)
    elif len(bbh_flist) > 1:
        raise IOError("Found multiple potential files for hot BB: %s" % bbh_flist)
    bbh = utils.readOMNIC(bbh_flist[0])
    logging.info("Found blackbody spectrum: %s", bbh_flist[0])

    # Load samples
    raw = load_sbm(fdir, ext=ext, lab=lab, notes_path=notes_path)
    labels  = raw['label']
    samples = [{'wn': raw['xaxis'], 'data': raw['sbm'][lbl]} for lbl in labels]
    n = len(samples)
    logging.info("Found %i samples in folder", n)

    # Axis consistency
    if len(bbc['wn']) != len(bbh['wn']) or not np.all(bbc['wn'] == bbh['wn']):
        raise IOError("Wavenumber axes do not match between warm and hot blackbodies")
    if len(raw['xaxis']) != len(bbc['wn']) or not np.all(raw['xaxis'] == bbc['wn']):
        raise IOError("Wavenumber axis mismatch between samples and blackbodies")
    wn = bbc['wn']

    # Auto-detect far-IR
    if wn.min() < 200 and wn.max() < 1700:
        logging.info("Far-IR spectrum detected with wn range %.0f-%.0f cm^-1", wn.min(), wn.max())
        fir = True

    # SBM mask
    sbm_diff = bbh["data"] - bbc["data"]
    idx = sbm_diff >= sbm_threshold
    n_total = len(idx)
    n_valid = int(idx.sum())
    logging.info(
        "SBM mask: %d/%d channels pass threshold %.4g  (diff range [%.4g, %.4g])",
        n_valid, n_total, sbm_threshold, sbm_diff.min(), sbm_diff.max(),
    )
    if n_valid == 0:
        raise ValueError(
            "SBM threshold %.4g rejected all %d channels "
            "(hot-warm diff range [%.4g, %.4g]). "
            "Check that bbhot is hotter than bbwarm, or lower sbm_threshold."
            % (sbm_threshold, n_total, sbm_diff.min(), sbm_diff.max())
        )
    if exclude is not None:
        idx = idx & ((wn < exclude[0]) | (wn > exclude[1]))
        if idx.sum() == 0:
            raise ValueError("After applying exclude range %s, no channels remain." % (exclude,))

    wn = wn[idx]

    # Ideal BB curves
    radc = utils.rad(wn, bbc_temp) * bb_emiss
    radh = utils.rad(wn, bbh_temp) * bb_emiss

    # IRF and instrument radiance (raw)
    # Ref: spectral_tools.dvrc::irf_ri, lines 320-325
    bbc_masked = bbc["data"][idx]
    bbh_masked = bbh["data"][idx]
    irf_raw = (radh - radc) / (bbh_masked - bbc_masked)
    ri_raw  = radc - bbc_masked * irf_raw

    if noise_free:
        # Ref: spectral_tools.dvrc::irf_ri, noise_free=1 (lines 341-396)
        # Fit Planck library to raw ri → noise-free instrument radiance
        nfri = _fit_planck_to_ri(ri_raw, wn)
        # Recompute one IRF per BB using nfri, then average (reduces noise by √2)
        irf1 = (radc - nfri) / bbc_masked
        irf2 = (radh - nfri) / bbh_masked
        irf  = (irf1 + irf2) / 2.0
        ri   = nfri
        logging.info("noise_free IRF: Planck-fit ri, averaged two-BB IRF")
    else:
        irf1 = irf2 = None
        irf  = irf_raw
        ri   = ri_raw

    # Calibrated radiance
    data = np.vstack([sample["data"][idx] * irf + ri for sample in samples])

    out: dict = {}
    out["xaxis"] = wn
    out["wl"]    = 1e4 / wn

    out["calib"] = {
        "irf":        irf,
        "irf_raw":    irf_raw,
        "irf1":       irf1,      # per-BB IRFs (None when noise_free=False)
        "irf2":       irf2,
        "ri":         ri,
        "ri_raw":     ri_raw,
        "noise_free": noise_free,
        "bbc_temp":   bbc_temp,
        "bbc_rad":    radc,
        "bbh_temp":   bbh_temp,
        "bbh_rad":    radh,
    }

    out["sbm"] = {"bbc": bbc["data"][idx], "bbh": bbh["data"][idx]}
    for i, sample in enumerate(samples):
        out["sbm"][labels[i]] = sample["data"][idx]

    out["rad"] = {
        "bbc": bbc["data"][idx] * irf + ri,
        "bbh": bbh["data"][idx] * irf + ri,
    }
    for i in range(n):
        out["rad"][labels[i]] = data[i, :]

    out["label"] = labels
    out["_fir"]  = fir
    out["_data"] = data   # stacked radiance; consumed by emcal, not part of public API

    # Embed per-sample measurement info so the viewer can plot it without re-reading the notes file.
    # Only populated for lab='nau' when notes were successfully loaded.
    if lab == "nau" and _notes_loaded and notes is not None:
        _ch_cols = ('channel_103', 'channel_104', 'channel_105', 'channel_106', 'channel_107')
        _notes_embed: dict = {'dtime': [], **{c: [] for c in _ch_cols}}

        def _dt_str(val) -> str:
            if pd.isna(val):
                return ''
            return val.isoformat() if hasattr(val, 'isoformat') else str(val)

        for lbl in labels:
            row = notes[notes['sample_name'] == lbl]
            if len(row) > 0:
                r0 = row.iloc[0]
                _notes_embed['dtime'].append(_dt_str(r0.get('dtime', '')))
                for col in _ch_cols:
                    val = r0.get(col, np.nan)
                    _notes_embed[col].append(float(val) if pd.notna(val) else float('nan'))
            else:
                _notes_embed['dtime'].append('')
                for col in _ch_cols:
                    _notes_embed[col].append(float('nan'))

        # BB resistance data (ch101/ch102) on a separate secondary-axis timeline
        _bb_names, _bb_dtimes, _bb_ch101, _bb_ch102 = [], [], [], []
        for _pat in (_bbc_pat, _bbh_pat):
            _row = notes[notes['sample_name'].str.contains(_pat, case=False, regex=True)]
            if len(_row) > 0:
                _r0 = _row.iloc[0]
                _bb_names.append(_r0['sample_name'])
                _bb_dtimes.append(_dt_str(_r0.get('dtime', '')))
                for _col, _lst in (('channel_101', _bb_ch101), ('channel_102', _bb_ch102)):
                    _v = _r0.get(_col, np.nan)
                    _lst.append(float(_v) if pd.notna(_v) else float('nan'))
        _notes_embed['bb_name']  = _bb_names
        _notes_embed['bb_dtime'] = _bb_dtimes
        _notes_embed['bb_ch101'] = _bb_ch101
        _notes_embed['bb_ch102'] = _bb_ch102

        out['notes'] = _notes_embed

    return out


# =============================================================================
# ============================== emcal ========================================
# =============================================================================
def emcal(
    fdir: str,
    bb1: float | tuple[float, float] | None = None,
    bb2: float | tuple[float, float] | None = None,
    ext: str = ".csv",
    method: str = "nem",
    bb_emiss: float = 0.995,
    sbm_threshold: float = 0.05,
    max_emiss: float = 1.0,
    wn_range: tuple[float, float] =  (500.0, 1700.0),
    lab: str = "nau",
    notes_path: str | None = None,
    plot: bool = False,
    plot_details: bool = False,
    save_plots: bool = False,
    save: bool = False,
    n_bb: int = 2,
    temp_halfwidth: float = 50.0,
    violation_weight: float = 5.0,
    violation_tol: float = 0.0,
    escalation_factor: float = 4.0,
    max_escalations: int = 4,
    fir: bool = False,
    exclude: tuple[float, float] | None = None,
    apply_dehyd: bool = False,
    noise_free: bool = True,
    downwelling_temps: dict[str, float] | None = None,
    on_missing_bb_temps: Callable[[], tuple[float, float]] | None = None,
    ow: bool = False,
) -> dict:
    """
    Perform emission calibration on a folder of FTIR spectra.

    Loads two blackbody spectra (warm and hot), derives the instrument
    response function, calibrates sample radiance, then retrieves emissivity
    using the chosen algorithm (NEM, MMD, or convex-hull LS fit).

    Parameters
    ----------
    fdir : str
        Path to the folder containing spectra.  ``~`` is expanded.
    bb1 : float or tuple[float, float] or None
        Warm blackbody temperature (K), or a (ch1_res, ch2_res) resistance
        pair to convert.  If None and lab='nau', read from the notes file.
    bb2 : float or tuple[float, float] or None
        Hot blackbody temperature (K), or resistance pair.  Same rules as
        *bb1*.
    ext : str
        File extension to search for (e.g. ``".csv"``).
    method : str
        Emissivity retrieval method: ``"nem"``, ``"mmd"``, ``"hullfit"``
        (convex-hull Planck mixture with strict upper-bound enforcement),
        ``"hullfit_linear"`` (fast closed-form two-temperature variant), or
        ``"alpha"`` (Alpha Residuals — mean-BT reference with max-emissivity
        rescaling).
    bb_emiss : float
        Assumed emissivity of the blackbody cavities.
    sbm_threshold : float
        Minimum (hot − warm) signal difference to retain a wavenumber.
    max_emiss : float
        Maximum allowed emissivity (normalisation ceiling).
    lab : str
        Lab convention for BB temperature conversion: ``"nau"``, ``"asu"``,
        or ``"swri"``.
    notes_path : str or None
        Explicit path to the measurement notes CSV file.  If None, the file
        is located automatically inside *fdir*.  Only used when ``lab='nau'``
        and BB temperatures are not supplied directly.
    plot : bool
        If True, display an emissivity summary plot.
    plot_details : bool
        If True, display a detailed radiance / model plot.
    save : bool
        If True, save the output dict to an HDF5 file in *fdir*.
    n_bb : int
        Number of blackbody components for ``method='hullfit'``.
    temp_halfwidth : float
        Half-width (K) of the auto-derived temperature search range for
        ``method='hullfit'``.  The range is centred on the peak brightness
        temperature within the fitting window.  Default 50 K.
    violation_weight : float
        Weight applied to channels where ``data > model`` in the hullfit
        violation-repair loop.  Default 5.0.
    violation_tol : float
        Fractional slack on the hullfit violation criterion.  Default 0.0
        (strict ``model ≥ data``).
    escalation_factor : float
        Weight multiplier applied when the hull set stalls.  Default 4.0.
    max_escalations : int
        Maximum weight-escalation attempts before giving up.  Default 4.
    fir : bool
        Force far-IR mode (overrides automatic detection).
    exclude : tuple[float, float] or None
        Wavenumber interval ``(wn_low, wn_high)`` to exclude from processing.
    apply_dehyd : bool
        If True, apply the :func:`dehyd` water vapour correction to each
        emissivity spectrum after retrieval.  Original emissivity is preserved
        in ``emiss_full[label]['dehyd']['emiss_orig']``.
    noise_free : bool
        If True (default), smooth the instrument response function with a
        Planck-library NNLS fit to the raw instrument radiance, then averages
        two per-BB IRF estimates to suppress noise.
        Ref: spectral_tools.dvrc::irf_ri, ``noise_free=1`` flag.
    downwelling_temps : dict[str, float] or None
        Per-sample downwelling (ambient) temperatures in °C, keyed by sample
        label.  Only used when ``lab='nau'``.  When provided, takes
        precedence over the ``channel_105`` column in the measurement info
        file (or acts as the sole source if the info file is absent).  When
        ``None`` and the info file cannot be found, the user is prompted
        interactively via stdin.

    Returns
    -------
    dict
        Nested result dictionary with keys ``wn``, ``wl``, ``calib``,
        ``sbm``, ``rad``, ``emiss``, ``emiss_full``, ``rad0``,
        ``sample_temps``.
    """
    if "~" in fdir:
        fdir = os.path.expanduser(fdir)
    fdir = os.path.abspath(fdir)

    _is_lab = lab in _LAB_INSTRUMENTS
    methods = {"nem":            "Normalized Emissivity",
               "mmd":            "Min/Max Difference",
               "alpha":          "Alpha Residuals",
               "hullfit_linear": "Convex Hull linear (n_bb=2)",
               "hullfit":        ("Convex Hull + NNLS (n_bb=3, lab)"
                                  if _is_lab else
                                  "Convex Hull linear (n_bb=2)")}

    if method == "hullfit":
        if _is_lab:
            logging.info("hullfit: lab instrument '%s' → emissivity_hullfit (n_bb=3)", lab)
        else:
            logging.info("hullfit: non-lab instrument '%s' → emissivity_hullfit_linear (n_bb=2)", lab)

    # Radiance calibration
    out = cal_rad(
        fdir,
        bb1=bb1, bb2=bb2, ext=ext,
        bb_emiss=bb_emiss, sbm_threshold=sbm_threshold,
        lab=lab, notes_path=notes_path,
        fir=fir, exclude=exclude,
        noise_free=noise_free,
        on_missing_bb_temps=on_missing_bb_temps,
    )

    wn     = out['xaxis']
    labels = out['label']
    n      = len(labels)
    fir    = out['_fir']
    data   = out.pop('_data')   # stacked radiance, shape (n, n_bands)

    # Build per-sample downwelling temperature lookup (K) from channel_105 (°C)
    # Ref: spectral_tools.dvrc::emcal2, line 3570
    if lab == "nau":
        notes, _ = _load_notes(notes_path if notes_path is not None else fdir)
        _notes_loaded = notes is not None
        if not _notes_loaded:
            logging.warning(
                "Measurement info file not found — downwelling temps unavailable from notes: %s",
                fdir,
            )
            if downwelling_temps is None:
                downwelling_temps = _prompt_downwelling_temps(labels)

        def _env_temp_k(label: str) -> float:
            if _notes_loaded:
                match = notes[notes["sample_name"] == label]
                if len(match) > 0:
                    return utils.c2k(float(match["channel_105"].values[0]))
                logging.warning(
                    "No channel_105 entry for '%s' — trying downwelling_temps", label
                )
            if downwelling_temps and label in downwelling_temps:
                return utils.c2k(downwelling_temps[label])
            logging.warning(
                "No downwelling temperature for '%s' — correction skipped", label
            )
            return 0.0
    else:
        def _env_temp_k(label: str) -> float:  # type: ignore[misc]
            return 0.0

    # Extract emissivity
    out["emiss"] = {}
    out["sample_temps"] = {}
    out["sample_t_wavenumber"] = {}
    out["emiss_full"] = {}
    out["rad0"] = {}
    for i in range(n):
        logging.info("Processing sample: %s", labels[i])
        if method == "nem":
            # Ref: spectral_tools.dvrc::emcal2, line 3570
            env_t = _env_temp_k(labels[i])
            if fir:
                em = emissivity_nem(wn, data[i, :], inst=lab,
                                    max_emiss=max_emiss, wn_range=wn_range,
                                    wn_range_cold=(400, 900),
                                    downwelling_t=env_t)
            else:
                em = emissivity_nem(wn, data[i, :], inst=lab,
                                    max_emiss=max_emiss, wn_range=wn_range,
                                    downwelling_t=env_t)
            out["emiss_full"][labels[i]] = em
            out["emiss"][labels[i]] = em["emiss"]
            out["rad0"][labels[i]] = em["rad_bb"]
        elif method == "mmd":
            em = emissivity_mmd(wn, data[i, :], max_emiss=max_emiss)
            out["emiss_full"][labels[i]] = em
            out["emiss"][labels[i]] = em["emiss"]
            out["rad0"][labels[i]] = em["rad0"]
        elif method == "alpha":
            env_t = _env_temp_k(labels[i])
            em = emissivity_alpha(
                wn, data[i, :], max_emiss=max_emiss, wn_range=wn_range,
                downwelling_t=env_t,
            )
            out["emiss_full"][labels[i]] = em
            out["emiss"][labels[i]] = em["emiss"]
            out["rad0"][labels[i]] = em["rad_bb"]
        elif method == "hullfit_linear":
            env_t = _env_temp_k(labels[i])
            em = emissivity_hullfit_linear(
                wn, data[i, :], max_emiss=max_emiss, wn_range=wn_range,
                downwelling_t=env_t,
                temp_halfwidth=temp_halfwidth,
            )
            out["emiss_full"][labels[i]] = em
            out["emiss"][labels[i]] = em["emiss"]
            out["rad0"][labels[i]] = em["rad_bb"]
        elif method == "hullfit":
            env_t = _env_temp_k(labels[i])
            if _is_lab:
                em = emissivity_hullfit(
                    wn, data[i, :], max_emiss=max_emiss, wn_range=wn_range,
                    n_bb=n_bb, downwelling_t=env_t,
                    temp_halfwidth=temp_halfwidth,
                    violation_weight=violation_weight,
                    violation_tol=violation_tol,
                    escalation_factor=escalation_factor,
                    max_escalations=max_escalations,
                )
            else:
                em = emissivity_hullfit_linear(
                    wn, data[i, :], max_emiss=max_emiss, wn_range=wn_range,
                    downwelling_t=env_t,
                    temp_halfwidth=temp_halfwidth,
                )
            out["emiss_full"][labels[i]] = em
            out["emiss"][labels[i]] = em["emiss"]
            out["rad0"][labels[i]] = em["rad_bb"]
        out["sample_temps"][labels[i]] = em["temp"]
        # Wavenumber at which the target temperature was determined (NEM only).
        # Mirrors DaVinci emcal2's sample_t_wavenumber field; NaN for hullfit
        # since the Planck mixture has no single peak channel.
        # Ref: spectral_tools.dvrc::emcal2, sample_t_wavenumber field
        if method == "nem":
            out["sample_t_wavenumber"][labels[i]] = (
                em["wn_t1"] if em["max_t1"] > em["threshold_t_warm"]
                else em["wn_t2"] if em["max_t2"] < em["threshold_t_cold"]
                else (em["wn_t1"] + em["wn_t2"]) / 2.0
            )
        else:
            out["sample_t_wavenumber"][labels[i]] = np.nan

        if method == "hullfit":
            # Calculate the weighted mean
            values = np.array(em["bb_temps"])
            weights = np.array(em["bb_fracs"])
            weighted_mean = np.average(values, weights=weights)

            # Calculate the weighted variance
            numerator = np.sum(weights * (values - weighted_mean) ** 2)
            denominator = np.sum(weights) * (len(values) - 1) / len(values)  # Bessel's correction for sample

            weighted_variance = numerator / denominator
            weighted_std = np.sqrt(weighted_variance)

            if weighted_std > 50:
                logging.info("Slope detected. Weighted standard deviation = %.1f K", weighted_std)

            else:
                logging.info("No significant slope detected. Weighted standard deviation = %.1f K", weighted_std)

        if apply_dehyd:
            d = dehyd(wn, out["emiss"][labels[i]])
            out["emiss"][labels[i]] = d["emiss"]
            out["emiss_full"][labels[i]]["dehyd"] = d

    # Stacked emissivity array ready for sma(): shape (n_samples, n_bands)
    out["data"] = np.stack([out["emiss"][label] for label in labels])
    out["label"] = [label for label in labels]
    out["method"] = methods[method]
    out["max_emiss"] = max_emiss

    # Plot results
    if plot or plot_details or save_plots:
        plot_emcal(out, plot_details=plot_details, save_plots=save_plots)


    # Save results
    if save:
        logging.info("Saving results ...")
        timestamp = '' if ow else datetime.now().strftime("%Y%m%d_%H%M%S")
        sep       = '_' if timestamp else ''
        fname_hdf = fdir + f"/emcal_results{sep}{timestamp}.hdf"
        fname_csv = fdir + f"/emcal_results{sep}{timestamp}.csv"
        utils.saveHDF(out, fname_hdf)
        logging.info("Saved HDF to %s", fname_hdf)
        utils.save_emcal_csv(out, fname_csv)
        logging.info("Saved CSV to %s", fname_csv)

    return out


# =============================================================================
# ================================ sma ========================================
# =============================================================================

# Alias for the NNLS solver already imported at module level
_scipy_nnls = nnls


# ---------------------------------------------------------------------------
# Private solver / statistics helpers
# ---------------------------------------------------------------------------

def _solve_nnls(
    E: np.ndarray,
    d: np.ndarray,
    n_constrained: int,
) -> np.ndarray:
    """
    Bounded least-squares solver with partial non-negativity constraints.

    Endmember matrix layout (must match the caller's construction order)::

        E = [ free minerals | forced minerals | BB | slope | atmospheric ]
              ←  constrained ≥ 0  →  ←  unconstrained (may be negative)  →

    Only the first *n_constrained* rows (free mineral endmembers) are
    required to be non-negative.  All remaining rows — forced minerals, BB,
    slope, and atmospheric — are unconstrained and may carry negative
    concentrations.  This mirrors DaVinci's ``lsqsn(..., num_no_rem, 0)``
    call where ``num_no_rem = forced_specs + bbval``.

    Ref: spectral_tools.dvrc::sma, ~lines 5031-5045 (lsqnn / lsqsn calls)

    Parameters
    ----------
    E : np.ndarray
        Endmember matrix, shape (n_endmembers, n_bands).
        Row order: ``[free minerals | forced | BB | slope | atmospheric]``.
    d : np.ndarray
        Observed spectrum, shape (n_bands,).
    n_constrained : int
        Number of rows at the head of *E* that are constrained ≥ 0
        (free mineral endmembers only).

    Returns
    -------
    np.ndarray
        Concentration vector in the same row order as *E*.
    """
    n_total = E.shape[0]

    if n_constrained == n_total:
        # All endmembers non-negative: plain NNLS (fastest path)
        c, _ = _scipy_nnls(E.T, d)
        return c

    if n_constrained == 0:
        # All unconstrained: plain OLS
        c, _, _, _ = np.linalg.lstsq(E.T, d, rcond=None)
        return c

    # Mixed case: use bounded LS (scipy lsq_linear, BVLS method).
    # Lower bounds: 0 for constrained rows, -inf for the rest.
    lb = np.empty(n_total)
    lb[:n_constrained] = 0.0
    lb[n_constrained:] = -np.inf
    ub = np.full(n_total, np.inf)

    result = lsq_linear(E.T, d, bounds=(lb, ub), method='bvls')
    return result.x


def _solve_ols_iter(
    E: np.ndarray,
    d: np.ndarray,
    n_forced: int,
) -> np.ndarray:
    """
    Unconstrained OLS with iterative removal of negative free-endmember
    concentrations (original DaVinci SMA algorithm).

    The last *n_forced* rows (forced minerals + atmospheric + BB) are never
    removed.  After each iteration all previously-zeroed free endmembers
    remain excluded, so
    the active set is monotonically shrinking and convergence is guaranteed.

    Ref: spectral_tools.dvrc::sma, lines 5054-5184

    Parameters
    ----------
    E : np.ndarray
        Endmember matrix, shape (n_endmembers, n_bands).
    d : np.ndarray
        Observed spectrum, shape (n_bands,).
    n_forced : int
        Number of forced endmembers at the tail of E.

    Returns
    -------
    np.ndarray
        Concentration vector, shape (n_endmembers,).
    """
    n_total = E.shape[0]
    n_free = n_total - n_forced

    c, _, _, _ = np.linalg.lstsq(E.T, d, rcond=None)

    while n_free > 0 and np.any(c[:n_free] < 0):
        # Permanently zero out negative free concentrations
        c[:n_free] = np.maximum(c[:n_free], 0.0)
        pos_mask = c[:n_free] > 0

        # Rebuild reduced system: positive free + all forced
        E_parts = []
        if pos_mask.any():
            E_parts.append(E[:n_free][pos_mask])
        if n_forced > 0:
            E_parts.append(E[n_free:])
        if not E_parts:
            break

        E_red = np.vstack(E_parts)
        c_red, _, _, _ = np.linalg.lstsq(E_red.T, d, rcond=None)

        # Map reduced solution back to full concentration vector
        c_new = c.copy()
        j = 0
        for i in range(n_free):
            if c[i] > 0:
                c_new[i] = c_red[j]
                j += 1
        c_new[n_free:] = c_red[j:]
        c = c_new

    return c


def _pixel_covariance(
    E_active: np.ndarray,
    c_active: np.ndarray,
    d: np.ndarray,
    nsamp: int,
) -> np.ndarray:
    """
    LS covariance matrix for the active endmembers at one pixel.

    Uses the standard OLS estimator covariance formula::

        C = (RSS / (nsamp - n_active)) * inv(E @ E.T)

    where RSS = d·d − c·(E @ d).

    Ref: spectral_tools.dvrc::sma, lines 5152-5179

    Parameters
    ----------
    E_active : np.ndarray
        Active endmember matrix, shape (n_active, n_bands).
    c_active : np.ndarray
        Concentrations for the active endmembers, shape (n_active,).
    d : np.ndarray
        Measured spectrum in the fitting range, shape (n_bands,).
    nsamp : int
        Number of spectral bands used in fitting.

    Returns
    -------
    np.ndarray
        Covariance matrix, shape (n_active, n_active).
    """
    n_active = E_active.shape[0]
    dof = nsamp - n_active
    if dof <= 0:
        return np.zeros((n_active, n_active))
    rss = float(d @ d - c_active @ (E_active @ d))
    sigma2 = rss / dof
    try:
        EET_inv = np.linalg.inv(E_active @ E_active.T)
    except np.linalg.LinAlgError:
        return np.zeros((n_active, n_active))
    return sigma2 * EET_inv


# ---------------------------------------------------------------------------
# Public helpers (translated from DaVinci)
# ---------------------------------------------------------------------------

def sort_cube(conc: np.ndarray) -> np.ndarray:
    """
    Return endmember indices sorted by concentration (descending) per pixel.

    In the DaVinci original this required a 70-line pixel loop because DV
    lacks native per-axis argsort.

    Ref: spectral_tools.dvrc::sort_cube, lines 5467-5539

    Parameters
    ----------
    conc : np.ndarray
        Concentration array, shape (..., n_endmembers).

    Returns
    -------
    np.ndarray
        Integer index array, same shape as *conc*.  Entry ``[..., k]``
        holds the 0-based index of the endmember with the (k+1)-th highest
        concentration at that pixel.
    """
    return np.argsort(conc, axis=-1)[..., ::-1].astype(np.intp)


def sum_group_conc(
    smaout: dict,
    forcedlib: dict | None = None,
) -> dict:
    """
    Sum endmember concentrations and propagate errors by mineral group.

    Group labels are taken from ``smaout['groups']``, which already
    contains labels for both mineral and forced endmembers.  The
    *forcedlib* parameter is accepted for interface compatibility with the
    DaVinci calling convention but is not required when groups are stored
    in *smaout*.

    Ref: spectral_tools.dvrc::sum_group_conc, lines 5543-5681

    Parameters
    ----------
    smaout : dict
        Output dictionary from :func:`sma`.  Required keys:
        ``conc``, ``bb``, ``groups``, ``E_fit``, ``measured_fit``,
        ``n_bands``.  The ``error`` key is optional; if absent only
        ``grouped_conc`` and ``grouped_labels`` are returned.
    forcedlib : dict or None
        Ignored when forced endmembers are already reflected in
        ``smaout['groups']`` (the default when called from :func:`sma`).
        Accepted for backward compatibility.

    Returns
    -------
    dict
        ``grouped_conc``   : np.ndarray (..., n_groups)
        ``grouped_labels`` : list[str]
        ``grouped_error``  : np.ndarray (..., n_groups)  — only when
                             ``smaout`` contains ``error``.
    """
    groups = list(smaout['groups'])          # per-endmember group names
    conc   = smaout['conc']                  # (..., n_endmembers)
    bb     = smaout['bb']                    # (...,)

    # Append BB
    groups.append('Black body')
    conc_full = np.concatenate(
        [conc, bb[..., np.newaxis]], axis=-1
    )                                        # (..., n_endmembers + 1)

    # Append Slope (only when the slope endmember was actually used in the fit)
    if smaout.get('has_slope', False):
        groups.append('Slope')
        conc_full = np.concatenate(
            [conc_full, smaout['slope'][..., np.newaxis]], axis=-1
        )

    # Unique groups in order of first appearance
    seen: dict[str, int] = {}
    for g in groups:
        if g not in seen:
            seen[g] = len(seen)
    unique_groups: list[str] = list(seen.keys())
    group_idx = np.array([seen[g] for g in groups], dtype=int)

    n_groups = len(unique_groups)

    # Sum concentrations per group: one-hot encode group membership then matmul.
    # group_onehot shape: (n_endmembers, n_groups); handles duplicate group indices.
    group_onehot = (
        group_idx[:, np.newaxis] == np.arange(n_groups)[np.newaxis, :]
    ).astype(float)
    grouped_conc = conc_full @ group_onehot   # (..., n_endmembers) @ (n_endmembers, n_groups)

    result: dict = {
        'grouped_conc':   grouped_conc,
        'grouped_labels': unique_groups,
    }

    if 'error' not in smaout:
        return result

    # Per-pixel error propagation via covariance sub-matrices
    # Ref: spectral_tools.dvrc::sum_group_conc, lines 5647-5671
    E_fit        = smaout['E_fit']          # (n_em_with_bb, n_fit_bands)
    measured_fit = smaout['measured_fit']   # (..., n_fit_bands)
    nsamp        = smaout['n_bands']

    spatial_shape = conc_full.shape[:-1]
    n_pix         = int(np.prod(spatial_shape)) if spatial_shape else 1
    conc_flat = conc_full.reshape(n_pix, -1)   # (n_pix, n_total)
    meas_flat = measured_fit.reshape(n_pix, -1) # (n_pix, n_fit_bands)
    grouped_error = np.zeros((n_pix, n_groups))

    for p in range(n_pix):
        d      = meas_flat[p]
        c      = conc_flat[p]
        active = c != 0.0
        if not active.any():
            continue
        E_active  = E_fit[active]
        c_active  = c[active]
        g_active  = group_idx[active]
        covm      = _pixel_covariance(E_active, c_active, d, nsamp)
        # group variance = sum of covariance sub-block for that group
        for gi in range(n_groups):
            in_group = g_active == gi
            if in_group.any():
                grouped_error[p, gi] = covm[np.ix_(in_group, in_group)].sum()

    grouped_error = np.sqrt(np.abs(grouped_error))
    result['grouped_error'] = grouped_error.reshape(
        spatial_shape + (n_groups,)
    )
    return result


# ---------------------------------------------------------------------------
# Slope ΔT inversion helper
# ---------------------------------------------------------------------------

def _invert_slope_delta_t(
    slope_normconc: np.ndarray,
    sample_ts: np.ndarray,
    wn: np.ndarray,
    fit_mask: np.ndarray,
    seed_dt: float = 5.0,
    n_grid: int = 500,
    dt_max: float = 50.0,
) -> np.ndarray:
    """
    Invert slope_normconc values to estimated ΔT (K) via an amplitude lookup.

    For each unique sample temperature (binned to 0.5 K), one amplitude-vs-ΔT
    lookup curve is built using fully vectorised Planck evaluations, then all
    spectra in that bin are inverted in a single ``np.interp`` call.  This
    keeps the cost proportional to the number of distinct temperatures, not
    the number of spectra — suitable for hyperspectral cubes.

    Parameters
    ----------
    slope_normconc : np.ndarray, shape (n_spectra,)
        Slope concentration as a percentage of mineral total (may be negative).
    sample_ts : np.ndarray, shape (n_spectra,)
        Per-sample centre temperature (K) used to build the lookup.
    wn : np.ndarray, shape (n_wn,)
        Full instrument wavenumber axis (cm⁻¹).
    fit_mask : np.ndarray, shape (n_wn,), bool
        True over the SMA fitting window (wn_range).
    seed_dt : float
        The ΔT (K) at which the slope endmember was computed.
    n_grid : int
        Number of ΔT points in the lookup grid.
    dt_max : float
        Upper bound of the ΔT grid (K).

    Returns
    -------
    np.ndarray, shape (n_spectra,)
        Estimated ΔT per spectrum.  Zero for negative or zero normconc.
        Clamped to dt_max for normconc that exceeds the lookup range.
    """
    dt_grid     = np.linspace(0.01, dt_max, n_grid)
    result      = np.zeros(len(slope_normconc), dtype=float)
    rounded_ts  = np.round(sample_ts * 2.0) / 2.0   # 0.5 K bins

    for T_c in np.unique(rounded_ts):
        mask = rounded_ts == T_c
        nc   = slope_normconc[mask]

        # Vectorised: (n_grid, n_wn) Planck ratio, normalised row-wise
        T_hi = T_c + dt_grid[:, np.newaxis] / 2.0
        T_lo = T_c - dt_grid[:, np.newaxis] / 2.0
        ratios  = utils.rad(wn, T_hi) / utils.rad(wn, T_lo)
        ratios /= ratios.max(axis=1, keepdims=True)
        amps    = 1.0 - ratios[:, fit_mask].min(axis=1)   # (n_grid,)

        seed_amp   = float(np.interp(seed_dt, dt_grid, amps))
        actual_amp = np.clip(nc, 0.0, None) / 100.0 * seed_amp
        result[mask] = np.interp(actual_amp, amps, dt_grid,
                                  left=0.0, right=dt_max)

    return result


# ---------------------------------------------------------------------------
# Main SMA function
# ---------------------------------------------------------------------------

def sma(
    em_data: np.ndarray | dict,
    em_xaxis: np.ndarray | None = None,
    endlib: dict | str | None = None,
    forcedlib: dict | str | None = None,
    atmlib: dict | str | None = None,
    wn_range: tuple[float, float] = (400.0, 1600.0),
    bb: bool = True,
    group: bool = True,
    exclude: list[int] | None = None,
    sort: bool = True,
    calc_errors: bool = True,
    notchco2: float = 0.0,
    nn: bool = True,
    forceall: bool = False,
    surface: bool = False,
    min_overlap: float = 0.8,
    plot: bool = False,
    save_plots: bool = False,
    plot_group: bool = False,
    plot_residual: bool = False,
    plot_error: bool = False,
    plot_cumulative: bool = False,
    plot_other: bool = False,
    plot_offset: float = 0.0,
    save: bool = False,
    save_path: str | None = None,
    slope: bool = False,
    sample_t: float | None = None,
    slope_seed_dt: float = 10.0,
) -> dict:
    """
    Spectral Mixture Analysis: decompose each spectrum in *em_data* into a
    linear combination of endmember spectra.

    Ref: spectral_tools.dvrc::sma, lines 4420-5463

    Parameters
    ----------
    em_data : np.ndarray or dict
        Either the emissivity data array directly, or an ``emcal`` output
        dict containing ``'data'`` (emissivity array) and ``'xaxis'``
        (wavenumber axis) keys — in which case *em_xaxis* may be omitted.

        When passing an array, accepted shapes are:

        * ``(n_bands,)``            — single spectrum
        * ``(n_spectra, n_bands)``  — 1-D array of spectra
        * ``(ny, nx, n_bands)``     — 2-D spatial cube

        Bands must be the last axis; wavenumbers correspond to *em_xaxis*.
    em_xaxis : np.ndarray or None
        Wavenumber axis (cm⁻¹) for *em_data*, shape (n_bands,).  Required
        when *em_data* is an array; ignored when *em_data* is a dict (the
        axis is taken from ``em_data['xaxis']`` instead).
    endlib : dict
        SpeclibViewerTIR album dict
        ``{spec_id: {'data', 'xaxis', 'category', 'sample_name', ...}}``.
        Each entry's spectrum is resampled to *em_xaxis* before fitting.
    forcedlib : dict or None
        Optional library of endmembers that are forced into every fit
        (e.g. a known mineral constituent) and are **NNLS-constrained**
        (concentration ≥ 0).  Same format as *endlib*.
    atmlib : dict or None
        Optional atmospheric endmember library.  Like *forcedlib* these
        are always included in the fit, but they are **unconstrained**
        (concentration may be negative) because the sign of an atmospheric
        contribution depends on the surface/atmosphere temperature contrast.
        When *atmlib* is provided, setting ``surface=True`` computes
        atmosphere-removed spectra.  Default ``None``.
    wn_range : tuple[float, float]
        ``(wn_low, wn_high)`` fitting window in cm⁻¹.  Only channels
        within this range are used for solving.  Default ``(400, 1600)``.
    bb : bool
        If True (default), append a flat unit spectrum (blackbody) as the
        final unconstrained endmember.
    group : bool
        If True (default), call :func:`sum_group_conc` to produce
        group-summed concentrations (requires ``'category'`` in library).
    exclude : list[int] or None
        Spec-IDs from *endlib* to exclude from the fit.  Their
        concentration slots in the output are filled with zero so the
        output shape always matches the full library.
    sort : bool
        If True (default), include ``'sort'`` in output: endmember indices
        ranked by concentration (descending) per pixel.
    calc_errors : bool
        If True (default), compute per-endmember 1-σ statistical errors
        from the LS covariance matrix.  Set False for large cubes.
    notchco2 : float
        CO₂ band removal.  ``0`` → disabled.  ``1`` → default half-width
        of 74.06 cm⁻¹ centred at 669 cm⁻¹.  Any other positive value →
        use that value as the half-width in cm⁻¹.
    nn : bool
        ``True`` (default) → non-negative least squares (NNLS, Lawson &
        Hanson 1974).  ``False`` → original unconstrained OLS with
        iterative negative-endmember removal.
    forceall : bool
        If True and ``nn=False``, skip negative removal (allow negative
        concentrations).  Ignored when ``nn=True``.
    surface : bool
        If True and *forcedlib* is provided, compute atmosphere-removed
        (``'rematm'``) and modeled-surface (``'modsur'``) spectra.
    min_overlap : float
        Minimum fraction of *wn_range* that an endmember spectrum must
        cover (after resampling to *em_xaxis*) to be accepted.
        Default 0.8.
    plot : bool
        If True, call :func:`~plot.plot_sma` to display one figure per
        sample after fitting.
    save_plots : bool (default = False) | passed on to plot_sma()
        If True, all samples are saved as PNG files to a default directory.
    plot_group : bool
        If True, pass ``group=True`` to :func:`~plot.plot_sma` so that
        concentrations are displayed by mineral group.  Requires
        ``group=True`` so grouped data is present in the output.
    plot_residual : bool
        If True, add a residual panel (measured − modeled) to each plot.
    plot_error : bool
        If True, show ``± X%`` error values on endmember overlay labels and
        pie chart entries.  Requires ``calc_errors=True``.
    plot_cumulative : bool
        If True, render the overlay as a cumulative stacked fill plot.
    plot_other : bool
        If True, aggregate all below-threshold endmembers into a single
        "Other" entry in the overlay.
    plot_offset : float
        Vertical offset (emissivity units) between successive endmember
        overlay spectra in individual (non-cumulative) mode.  Default 0.0.
    save : bool
        If True, save the output dict to an HDF5 file and a CSV file.
    save_path : str or None
        Directory in which to write the output files.  If None, the current
        working directory is used.
    slope : bool
        If True, generate a synthetic "spectral slope" endmember from a 50/50
        blackbody radiance mixture at the two *slope_t* temperatures, then
        derive its emissivity as ``L_mix / B(T_max)`` (pinned at 1.0 at the
        peak channel within *wn_range*).  The slope endmember is excluded from
        mineral normalisation and reported in ``slope_normconc``.
        Default False.
    sample_t : float or None
        Centre temperature (K) for the slope endmember ``B(T+seed/2) /
        B(T-seed/2)``.  When None (default) and *em_data* is an emcal output
        dict, the mean of ``em_data['sample_temps']`` is used automatically.
        When None and *em_data* is a raw array, falls back to 300 K with a
        warning.
    slope_seed_dt : float
        Half-width of the seed temperature separation (K).  The slope
        endmember is ``B(sample_t + seed/2) / B(sample_t - seed/2)``,
        normalised to 1.0 at its spectral peak.  Default 10.0 K.

    Returns
    -------
    dict
        Result dictionary with the following keys:

        ``measured``       : np.ndarray (..., n_bands_notched) — input spectra
        ``modeled``        : np.ndarray (..., n_bands_notched) — reconstructed
        ``rms``            : np.ndarray (...) — per-pixel RMS residual
        ``conc``           : np.ndarray (..., n_lib) — raw concentrations
                             (zeros at excluded positions)
        ``normconc``       : np.ndarray (..., n_lib) — normalised to 100 %
                             (mineral endmembers only; BB and slope excluded)
        ``bb``             : np.ndarray (...) — blackbody fraction
        ``bb_normconc``    : np.ndarray (...) — BB as % of grand total
        ``slope``            : np.ndarray (...) — slope fraction (if slope=True)
        ``slope_normconc``   : np.ndarray (...) — slope as % of minerals (if slope=True)
        ``delta_t_estimated``: np.ndarray (...) — estimated ΔT (K) inverted from
                               slope_normconc via amplitude lookup (if slope=True)
        ``error``          : np.ndarray (..., n_lib) — 1-σ errors (if requested)
        ``bberror``        : np.ndarray (...) — blackbody error (if requested)
        ``slopeerror``     : np.ndarray (...) — slope error (if requested)
        ``normerror``      : np.ndarray (..., n_lib) — normalised errors
        ``labels``         : list[str] — mineral endmember labels
                             (length n_minerals_orig + n_forced_lib)
        ``groups``         : list[str] — mineral endmember group names
        ``atm_conc``       : np.ndarray (..., n_atm) — atmospheric concentrations
                             (may be negative); absent when atmlib is None
        ``atm_labels``     : list[str] — atmospheric endmember labels;
                             absent when atmlib is None
        ``xaxis``          : np.ndarray — wavenumber axis (post-notch)
        ``n_bands``        : int — number of bands in the fitting window
        ``wn_range``       : tuple[float, float]
        ``algorithm``      : str
        ``notchco2``       : dict or None
        ``excluded``       : list[int] or None — spec-IDs that were excluded
        ``sort``           : np.ndarray (..., n_lib) — sorted indices (if requested)
        ``grouped``        : dict — group sums (if ``group=True``)
        ``modsur``         : np.ndarray — modelled surface (if ``surface=True``)
        ``rematm``         : np.ndarray — atm-removed emissivity (if ``surface=True``)

    Raises
    ------
    ValueError
        If any endmember covers less than *min_overlap* of *wn_range*, if
        no channels fall within *wn_range*, if the endmember matrix is
        rank-deficient, or if any NaN values remain in the fitting window
        after resampling.
    """
    # ------------------------------------------------------------------
    # 0. Input normalisation
    # ------------------------------------------------------------------
    # Load from file if paths are given
    if isinstance(em_data, str):
        em_data = utils.readHDF(em_data)
    if isinstance(endlib, str):
        endlib = _to_album(utils.readHDF(endlib))
    elif isinstance(endlib, dict):
        endlib = _to_album(endlib)
    if isinstance(forcedlib, str):
        forcedlib = _to_album(utils.readHDF(forcedlib))
    elif isinstance(forcedlib, dict):
        forcedlib = _to_album(forcedlib)
    if isinstance(atmlib, str):
        atmlib = _to_album(utils.readHDF(atmlib))
    elif isinstance(atmlib, dict):
        atmlib = _to_album(atmlib)

    # Accept an emcal output dict in place of separate em_data / em_xaxis
    # Save the dict reference before it is overwritten so slope temperature
    # resolution and per-sample ΔT inversion can access sample_temps later.
    _emcal_dict: dict | None = em_data if isinstance(em_data, dict) else None

    sample_labels: list[str] | None = None
    if isinstance(em_data, dict):
        if endlib is None:
            raise ValueError("endlib must be supplied when em_data is an emcal dict")
        em_xaxis = np.asarray(em_data['xaxis'], dtype=np.float64)
        if 'label' in em_data:
            sample_labels = list(em_data['label'])
        em_data  = np.asarray(em_data['data'],  dtype=np.float64)
    else:
        if em_xaxis is None:
            raise ValueError("em_xaxis is required when em_data is an array")
        if endlib is None:
            raise ValueError("endlib must be supplied")
        em_data  = np.asarray(em_data,  dtype=np.float64)
        em_xaxis = np.asarray(em_xaxis, dtype=np.float64)

    if em_data.ndim == 1:
        em_data      = em_data[np.newaxis, :]
        spatial_shape: tuple[int, ...] = ()
    elif em_data.ndim == 2:
        spatial_shape = (em_data.shape[0],)
    elif em_data.ndim == 3:
        ny, nx, _    = em_data.shape
        spatial_shape = (ny, nx)
        em_data      = em_data.reshape(-1, em_data.shape[-1])
    else:
        raise ValueError("em_data must be 1-D, 2-D, or 3-D")

    n_spectra = em_data.shape[0]

    if em_data.shape[1] != em_xaxis.shape[0]:
        raise ValueError(
            f"em_data has {em_data.shape[1]} bands but em_xaxis has "
            f"{em_xaxis.shape[0]} elements"
        )

    # ------------------------------------------------------------------
    # 1. Resample and validate the endmember library
    # ------------------------------------------------------------------
    wn_lo, wn_hi = wn_range
    fit_width    = wn_hi - wn_lo

    def _build_em_rows(lib: dict, lib_name: str) -> tuple[
        np.ndarray, list[str], list[str]
    ]:
        """Resample all entries in *lib* to em_xaxis; return (E, labels, groups)."""
        rows, lbs, grps = [], [], []
        for sid, entry in lib.items():
            src_x = np.asarray(entry['xaxis'], dtype=np.float64)
            src_d = np.asarray(entry['data'],  dtype=np.float64)

            # Overlap check in the fitting window
            ovlp_lo = max(em_xaxis.min(), src_x.min())
            ovlp_hi = min(em_xaxis.max(), src_x.max())
            use_lo  = max(ovlp_lo, wn_lo)
            use_hi  = min(ovlp_hi, wn_hi)
            if use_hi <= use_lo:
                raise ValueError(
                    f"{lib_name} spec_id {sid} has no overlap with "
                    f"wn_range {wn_range}"
                )
            frac = (use_hi - use_lo) / fit_width
            if frac < min_overlap:
                raise ValueError(
                    f"{lib_name} spec_id {sid} covers only "
                    f"{100*frac:.1f}% of wn_range {wn_range} "
                    f"(min_overlap={100*min_overlap:.0f}%)"
                )

            rows.append(resample_spectrum(src_x, src_d, em_xaxis))
            lbs.append(
                f"{entry.get('sample_name', 'Unknown')} "
                f"{entry.get('spec_id', sid)}"
            )
            grps.append(entry.get('category', 'Unknown'))
        return np.stack(rows), lbs, grps

    E_minerals, mineral_labels, mineral_groups = _build_em_rows(endlib, 'endlib')
    all_mineral_ids = list(endlib.keys())
    n_minerals_orig = E_minerals.shape[0]

    # Forced mineral endmembers (NNLS-constrained, concentration ≥ 0)
    n_forced_lib = 0
    E_forced_full: np.ndarray | None = None
    forced_labels: list[str] = []
    forced_groups: list[str] = []

    if forcedlib is not None:
        E_forced_full, forced_labels, forced_groups = _build_em_rows(
            forcedlib, 'forcedlib'
        )
        n_forced_lib = E_forced_full.shape[0]

    # Atmospheric endmembers (lstsq-unconstrained, concentration may be negative)
    n_atm_lib = 0
    E_atm_full: np.ndarray | None = None
    atm_labels: list[str] = []
    atm_groups: list[str] = []

    if atmlib is not None:
        E_atm_full, atm_labels, atm_groups = _build_em_rows(atmlib, 'atmlib')
        n_atm_lib = E_atm_full.shape[0]

    # ------------------------------------------------------------------
    # 2. CO₂ notch removal
    # Ref: spectral_tools.dvrc::sma, lines 4687-4743
    # ------------------------------------------------------------------
    notchco2_info: dict | None = None
    keep_mask = np.ones(em_xaxis.shape[0], dtype=bool)

    if notchco2 != 0.0:
        co2_wn  = 669.0
        wco2    = 74.06 if notchco2 == 1 else float(notchco2)
        notch_lo = co2_wn - wco2
        notch_hi = co2_wn + wco2
        keep_mask = (em_xaxis < notch_lo) | (em_xaxis > notch_hi)
        notchco2_info = {
            'wco2':     wco2,
            'co2_low':  notch_lo,
            'co2_high': notch_hi,
        }
        logging.info(
            "CO₂ notch applied: %.1f–%.1f cm⁻¹ (%d channels removed)",
            notch_lo, notch_hi, (~keep_mask).sum(),
        )

    xaxis_notched    = em_xaxis[keep_mask]
    em_data_notched  = em_data[:, keep_mask]
    E_min_notched    = E_minerals[:, keep_mask]
    E_forced_notched = (
        E_forced_full[:, keep_mask] if E_forced_full is not None else None
    )
    E_atm_notched    = (
        E_atm_full[:, keep_mask] if E_atm_full is not None else None
    )

    # ------------------------------------------------------------------
    # 3. Endmember exclusion
    # Ref: spectral_tools.dvrc::sma, lines 4854-4875
    # ------------------------------------------------------------------
    excluded_info: list[int] | None = None
    excluded_positions: list[int] = []   # positional (0-based) in mineral list
    kept_positions:    list[int] = list(range(n_minerals_orig))  # updated if exclude

    if exclude is not None:
        exclude_set = set(exclude)
        excluded_positions = [
            i for i, sid in enumerate(all_mineral_ids) if sid in exclude_set
        ]
        unknown = exclude_set - set(all_mineral_ids)
        if unknown:
            logging.warning(
                "exclude contains spec_ids not in endlib: %s", unknown
            )
        keep_em = np.ones(n_minerals_orig, dtype=bool)
        for pos in excluded_positions:
            keep_em[pos] = False

        E_min_notched  = E_min_notched[keep_em]
        mineral_labels = [l for l, k in zip(mineral_labels, keep_em) if k]
        mineral_groups = [g for g, k in zip(mineral_groups, keep_em) if k]
        excluded_info  = [all_mineral_ids[p] for p in excluded_positions]
        logging.info("Excluded %d endmember(s): %s", len(excluded_positions), excluded_info)

    n_mineral_active = E_min_notched.shape[0]

    # ------------------------------------------------------------------
    # 4. Validate zero-spectrum check
    # Ref: spectral_tools.dvrc::sma, lines 4680-4684
    # ------------------------------------------------------------------
    if np.any(np.all(E_min_notched == 0, axis=1)):
        raise ValueError(
            "One or more endmembers are all-zero after resampling. "
            "Check the library spectra."
        )

    # ------------------------------------------------------------------
    # 4.5. Synthetic slope endmember
    # slope_emiss = B(T_center + seed/2) / B(T_center - seed/2), normalised
    # to 1.0 at its peak over the full instrument spectral range (em_xaxis).
    # T_center is resolved from: explicit sample_t > emcal sample_temps > 300 K.
    # ------------------------------------------------------------------
    slope_int   = 0
    slope_emiss: np.ndarray | None = None
    _slope_t_center: float = 300.0          # resolved below when slope=True

    if slope:
        if sample_t is not None:
            _slope_t_center = float(sample_t)
        elif _emcal_dict is not None and 'sample_temps' in _emcal_dict:
            _temps = [float(v) for v in _emcal_dict['sample_temps'].values()
                      if v is not None]
            if _temps:
                _slope_t_center = float(np.mean(_temps))
            else:
                logging.warning("slope: emcal sample_temps is empty; using 300 K")
        else:
            logging.warning(
                "slope: sample_t not provided and no emcal temperature available; "
                "using 300 K — pass sample_t explicitly for accurate results"
            )

        T_low  = _slope_t_center - slope_seed_dt / 2.0
        T_high = _slope_t_center + slope_seed_dt / 2.0
        _ratio      = utils.rad(em_xaxis, T_high) / utils.rad(em_xaxis, T_low)
        slope_emiss = _ratio / float(_ratio.max())
        slope_int   = 1
        logging.info(
            "Slope endmember: B(%.2f K) / B(%.2f K)  "
            "(T_center=%.1f K, seed_ΔT=%.1f K)  range=[%.4f, 1.0000]",
            T_high, T_low, _slope_t_center, slope_seed_dt, float(slope_emiss.min()),
        )

    # ------------------------------------------------------------------
    # 5. Concatenate all endmembers and add blackbody row
    # Order: [free minerals | forced minerals | BB | slope | atmospheric]
    #        ←  NNLS (≥ 0)  →  ←  unconstrained (may be negative)  →
    # Forced minerals, BB, and slope are never removed (mirrors DaVinci's
    # num_no_rem group); atmospheric endmembers are also unconstrained.
    # Ref: spectral_tools.dvrc::sma, lines 4877-4897
    # ------------------------------------------------------------------
    parts = [E_min_notched]
    if E_forced_notched is not None:
        parts.append(E_forced_notched)      # type: ignore[arg-type]

    # Labels and groups cover mineral endmembers only (free + forced).
    # Atmospheric endmembers are tracked separately in atm_labels.
    all_labels = mineral_labels + forced_labels
    all_groups = mineral_groups + forced_groups

    # Number of mineral endmembers in the active (post-exclusion) solve
    n_min_active_total = n_mineral_active + n_forced_lib

    bb_int = 1 if bb else 0

    if bb:
        parts.append(np.ones((1, xaxis_notched.shape[0])))

    if slope and slope_emiss is not None:
        parts.append(slope_emiss[keep_mask][np.newaxis, :])

    if E_atm_notched is not None:
        parts.append(E_atm_notched)         # type: ignore[arg-type]

    E_full = np.vstack(parts)                   # (n_total_em, n_notched_bands)
    n_total_em = E_full.shape[0]

    # ------------------------------------------------------------------
    # 6. Wavenumber range crop → fitting window
    # Ref: spectral_tools.dvrc::sma, lines 4906-4958
    # ------------------------------------------------------------------
    fit_ch_mask = (xaxis_notched >= wn_lo) & (xaxis_notched <= wn_hi)
    ch_indices  = np.where(fit_ch_mask)[0]

    if ch_indices.size == 0:
        raise ValueError(
            f"No channels remain within wn_range {wn_range} after "
            f"CO₂ notch removal."
        )

    ch_start   = int(ch_indices[0])
    ch_end     = int(ch_indices[-1]) + 1
    xaxis_fit  = xaxis_notched[ch_start:ch_end]
    E_fit      = E_full[:, ch_start:ch_end]     # (n_total_em, n_fit_bands)
    data_fit   = em_data_notched[:, ch_start:ch_end]
    n_fit_bands = xaxis_fit.shape[0]

    # Check for NaN in the fitting window
    nan_em = np.any(np.isnan(E_fit), axis=1)
    if nan_em.any():
        bad = np.where(nan_em)[0].tolist()
        raise ValueError(
            f"Endmember row(s) {bad} contain NaN within wn_range {wn_range}. "
            f"Ensure spectra cover the fitting window."
        )

    if n_total_em > n_fit_bands:
        raise ValueError(
            f"More endmembers ({n_total_em}) than fitting bands ({n_fit_bands})."
        )

    logging.info(
        "SMA: %d spectra | %d endmembers "
        "(%d free mineral, %d forced mineral, %d atm, %d BB, %d slope) | "
        "%d fitting bands (%.0f–%.0f cm⁻¹) | algorithm: %s",
        n_spectra, n_total_em,
        n_mineral_active, n_forced_lib, n_atm_lib, bb_int, slope_int,
        n_fit_bands, xaxis_fit[0], xaxis_fit[-1],
        "NNLS" if nn else ("OLS+iter" if not forceall else "OLS"),
    )

    # ------------------------------------------------------------------
    # 7. Core solve: one concentration vector per spectrum
    # Ref: spectral_tools.dvrc::sma, lines 5017-5184
    # ------------------------------------------------------------------
    conc_all = np.zeros((n_spectra, n_total_em))

    # OLS forced count: forced minerals + atm + BB + slope (never removed from iteration)
    n_ols_forced = n_forced_lib + n_atm_lib + bb_int + slope_int

    for i in range(n_spectra):
        d = data_fit[i]
        if nn:
            conc_all[i] = _solve_nnls(E_fit, d, n_mineral_active)
        elif not forceall:
            conc_all[i] = _solve_ols_iter(E_fit, d, n_ols_forced)
        else:
            c, _, _, _ = np.linalg.lstsq(E_fit.T, d, rcond=None)
            conc_all[i] = c

    # ------------------------------------------------------------------
    # 8. Statistical errors
    # Ref: spectral_tools.dvrc::sma, lines 5152-5180
    # ------------------------------------------------------------------
    err_all = np.zeros((n_spectra, n_total_em)) if calc_errors else None
    last_covm: np.ndarray | None = None

    if calc_errors:
        assert err_all is not None
        for i in range(n_spectra):
            d      = data_fit[i]
            c      = conc_all[i]
            active = c != 0.0
            if not active.any():
                continue
            E_active = E_fit[active]
            c_active = c[active]
            covm     = _pixel_covariance(E_active, c_active, d, n_fit_bands)
            last_covm = covm
            err_diag = np.sqrt(np.abs(np.diag(covm)))
            err_all[i, active] = err_diag

    # ------------------------------------------------------------------
    # 9. Reconstruction and RMS
    # Ref: spectral_tools.dvrc::sma, lines 5190-5309
    # ------------------------------------------------------------------
    recon_fit  = conc_all @ E_fit         # (n_spectra, n_fit_bands)  for RMS
    recon_full = conc_all @ E_full        # (n_spectra, n_notched_bands)

    rms_flat = np.sqrt(
        np.mean((data_fit - recon_fit) ** 2, axis=-1)
    )                                     # (n_spectra,)

    # ------------------------------------------------------------------
    # 10. Optional surface / atmosphere-removal
    # Ref: spectral_tools.dvrc::sma, lines 5197-5302
    # ------------------------------------------------------------------
    modsur_flat: np.ndarray | None = None
    rematm_flat: np.ndarray | None = None

    if surface and n_atm_lib > 0:
        # Surface spectrum = mineral (free + forced) + BB, excluding atmospheric.
        # In the new layout [minerals | BB | atm], these are the first
        # n_min_active_total + bb_int rows.
        surf_idx = list(range(n_min_active_total + bb_int))
        E_surf      = E_full[surf_idx]
        c_surf      = conc_all[:, surf_idx]
        modsur_flat = c_surf @ E_surf

        residual_full = em_data_notched - recon_full
        rematm_flat   = modsur_flat + residual_full

        # Add summed atmospheric concentration as a scalar offset per pixel
        atm_start  = n_min_active_total
        atm_end    = atm_start + n_atm_lib
        atm_sum    = conc_all[:, atm_start:atm_end].sum(axis=-1, keepdims=True)
        modsur_flat = modsur_flat + atm_sum
        rematm_flat = rematm_flat + atm_sum

    # ------------------------------------------------------------------
    # 11. BB and atmospheric extraction
    # Layout in conc_all: [minerals (0:n_min) | BB (n_min) | atm (n_min+bb_int:)]
    # Ref: spectral_tools.dvrc::sma, lines 5250-5265
    # ------------------------------------------------------------------
    # Mineral concentrations (free + forced, post-exclusion width)
    conc_minerals = conc_all[:, :n_min_active_total]
    err_minerals  = err_all[:, :n_min_active_total] if err_all is not None else None

    # BB sits immediately after minerals
    if bb:
        bb_flat    = conc_all[:, n_min_active_total]
        bberr_flat = err_all[:, n_min_active_total] if err_all is not None else None
    else:
        bb_flat    = np.zeros(n_spectra)
        bberr_flat = np.zeros(n_spectra) if calc_errors else None

    # Slope sits immediately after BB
    if slope:
        slope_pos     = n_min_active_total + bb_int
        slope_flat    = conc_all[:, slope_pos]
        slopeerr_flat = err_all[:, slope_pos] if err_all is not None else None
    else:
        slope_flat    = np.zeros(n_spectra)
        slopeerr_flat = np.zeros(n_spectra) if calc_errors else None

    # Atmospheric concentrations (may be negative) are the remaining tail
    atm_start = n_min_active_total + bb_int + slope_int
    if n_atm_lib > 0:
        atm_flat = conc_all[:, atm_start:]
        atm_err  = err_all[:, atm_start:] if err_all is not None else None
    else:
        atm_flat = None
        atm_err  = None

    # Alias used by the rest of the function
    conc_no_bb = conc_minerals
    err_no_bb  = err_minerals

    # ------------------------------------------------------------------
    # 12. Expand excluded endmembers back to original library size
    # Ref: spectral_tools.dvrc::sma, lines 5317-5378
    # ------------------------------------------------------------------
    n_out_lib = n_minerals_orig + n_forced_lib   # final output width

    if exclude is not None and excluded_positions:
        full_conc = np.zeros((n_spectra, n_out_lib))
        full_err  = (
            np.zeros((n_spectra, n_out_lib)) if err_no_bb is not None else None
        )
        kept_positions = [
            i for i in range(n_minerals_orig)
            if i not in set(excluded_positions)
        ]   # overrides the initialised value
        full_conc[:, kept_positions] = conc_no_bb[:, :n_mineral_active]
        if n_forced_lib > 0:
            full_conc[:, n_minerals_orig:] = conc_no_bb[:, n_mineral_active:]
        if full_err is not None:
            full_err[:, kept_positions] = err_no_bb[:, :n_mineral_active]
            if n_forced_lib > 0:
                full_err[:, n_minerals_orig:] = err_no_bb[:, n_mineral_active:]
        conc_no_bb = full_conc
        err_no_bb  = full_err

        # Rebuild labels / groups to match the expanded output
        full_mineral_labels = [
            f"{endlib[sid].get('sample_name','Unknown')} "
            f"{endlib[sid].get('spec_id', sid)}"
            for sid in all_mineral_ids
        ]
        full_mineral_groups = [
            endlib[sid].get('category', 'Unknown')
            for sid in all_mineral_ids
        ]
        all_labels = full_mineral_labels + forced_labels
        all_groups = full_mineral_groups + forced_groups

    # Normalisation.
    # normconc       = mineral_i / sum(minerals) * 100  [minerals only, sum to 100%]
    # bb_normconc    = c_bb      / sum(minerals) * 100  [may be negative]
    # slope_normconc = c_slope   / sum(minerals) * 100  [may be negative]
    #
    # Using sum(minerals) as the common denominator keeps bb_normconc and
    # slope_normconc on the same scale as normconc and avoids divide-by-zero
    # when BB or slope goes negative (which happens now that they are
    # unconstrained in the NNLS solver).
    total_conc   = np.sum(conc_no_bb, axis=-1, keepdims=True)  # (n_spectra, 1)
    total_mineral = total_conc[..., 0]                          # (n_spectra,)

    normconc = np.where(
        total_conc > 0, conc_no_bb / total_conc * 100.0, 0.0
    )

    bb_normconc = np.where(
        total_mineral > 0, bb_flat / total_mineral * 100.0, 0.0
    )

    slope_normconc_arr = np.where(
        total_mineral > 0, slope_flat / total_mineral * 100.0, 0.0
    )

    normerror: np.ndarray | None = None
    if err_no_bb is not None:
        normerror = np.where(
            total_conc > 0, err_no_bb / total_conc * 100.0, 0.0
        )

    # ------------------------------------------------------------------
    # 13. Sort endmember indices by concentration per pixel
    # Ref: spectral_tools.dvrc::sma, lines 5311-5377
    # ------------------------------------------------------------------
    sort_arr: np.ndarray | None = None
    if sort:
        sort_arr = sort_cube(conc_no_bb)

    # ------------------------------------------------------------------
    # 13b. ΔT inversion from slope_normconc
    # Build an amplitude-vs-ΔT lookup per unique sample temperature and
    # invert slope_normconc to an estimated ΔT.  Grouped by temperature
    # (0.5 K bins) so the lookup is built once per bin regardless of how
    # many spectra share it — efficient for hyperspectral cubes.
    # ------------------------------------------------------------------
    delta_t_flat: np.ndarray = np.zeros(n_spectra)
    if slope:
        if _emcal_dict is not None and 'sample_temps' in _emcal_dict and sample_labels:
            _per_sample_ts = np.array([
                float(_emcal_dict['sample_temps'].get(lbl, _slope_t_center))
                for lbl in sample_labels
            ])
        else:
            _per_sample_ts = np.full(n_spectra, _slope_t_center)

        _fit_mask_full = (em_xaxis >= wn_lo) & (em_xaxis <= wn_hi)
        delta_t_flat   = _invert_slope_delta_t(
            slope_normconc_arr, _per_sample_ts,
            em_xaxis, _fit_mask_full, slope_seed_dt,
        )

    # ------------------------------------------------------------------
    # 14. Reshape all outputs to original spatial dimensions
    # ------------------------------------------------------------------
    def _rs(arr: np.ndarray) -> np.ndarray:
        """Reshape leading n_spectra axis back to spatial_shape."""
        if not spatial_shape:
            return arr[0]
        return arr.reshape(spatial_shape + arr.shape[1:])

    out_measured        = _rs(em_data_notched)
    out_modeled         = _rs(recon_full)
    out_rms             = _rs(rms_flat)
    out_conc            = _rs(conc_no_bb)
    out_normconc        = _rs(normconc)
    out_bb              = _rs(bb_flat)
    out_bberr           = (_rs(bberr_flat)     if bberr_flat     is not None else None)
    out_slope           = _rs(slope_flat)
    out_slopeerr        = (_rs(slopeerr_flat)   if slopeerr_flat  is not None else None)
    out_slope_normconc  = _rs(slope_normconc_arr)
    out_err             = (_rs(err_no_bb)       if err_no_bb      is not None else None)
    out_normerr         = (_rs(normerror)       if normerror      is not None else None)
    out_modsur          = (_rs(modsur_flat)     if modsur_flat    is not None else None)
    out_rematm          = (_rs(rematm_flat)     if rematm_flat    is not None else None)
    out_sort            = (_rs(sort_arr)        if sort_arr       is not None else None)
    out_atm_conc        = (_rs(atm_flat)        if atm_flat       is not None else None)
    out_bb_normconc     = _rs(bb_normconc)
    out_delta_t         = _rs(delta_t_flat)

    # ------------------------------------------------------------------
    # 15. Surface correction for non-unit concentration sums
    # Ref: spectral_tools.dvrc::sma, lines 5407-5412
    # ------------------------------------------------------------------
    if out_modsur is not None and out_rematm is not None:
        corr = (
            1.0
            - out_bb[..., np.newaxis]
            - np.sum(out_conc, axis=-1, keepdims=True)
        )
        out_modsur  = out_modsur  + corr
        out_rematm  = out_rematm  + corr

    # ------------------------------------------------------------------
    # 16. Build E_fit for sum_group_conc
    #     sum_group_conc works on [conc | bb] (mineral concentrations only),
    #     so E_fit_out must have shape (n_out_lib + bb_int, n_fit_bands) —
    #     atmospheric endmembers are excluded from the stored E_fit.
    # ------------------------------------------------------------------
    # E_fit rows 0 : n_min_active_total + bb_int are minerals + BB (contiguous).
    # Atmospheric rows sit beyond that and are excluded from the stored E_fit.
    if exclude is not None and excluded_positions:
        E_fit_out = np.zeros((n_out_lib + bb_int + slope_int, n_fit_bands))
        E_fit_out[kept_positions]                = E_fit[:n_mineral_active]
        if n_forced_lib > 0:
            E_fit_out[n_minerals_orig:n_out_lib] = E_fit[n_mineral_active:n_min_active_total]
        if bb:
            E_fit_out[n_out_lib] = E_fit[n_min_active_total]
        if slope:
            E_fit_out[n_out_lib + bb_int] = E_fit[n_min_active_total + bb_int]
    else:
        E_fit_out = E_fit[:n_min_active_total + bb_int + slope_int]

    # measured_fit for sum_group_conc covariance (keep in flat shape then reshape)
    out_measured_fit = _rs(data_fit)

    # ------------------------------------------------------------------
    # 17. Algorithm tag
    # ------------------------------------------------------------------
    if nn:
        algo = "Nonnegative least squares (NNLS)"
    elif forceall:
        algo = "Unconstrained least squares (OLS)"
    else:
        algo = "OLS with iterative negative-endmember removal"

    # ------------------------------------------------------------------
    # 18. Assemble output dictionary
    # ------------------------------------------------------------------
    smaout: dict = {
        # Spectra
        'measured':      out_measured,
        'modeled':       out_modeled,
        'rms':           out_rms,
        # Concentrations
        'conc':          out_conc,
        'normconc':      out_normconc,      # each mineral as % of sum(minerals)
        'bb':            out_bb,
        'bb_normconc':   out_bb_normconc,   # BB as % of (minerals + BB + slope)
        'slope':         out_slope,         # zero array when slope=False
        'slope_normconc': out_slope_normconc,  # slope as % of (minerals + slope)
        # Labels / groups  (length = n_minerals_orig + n_forced_lib)
        'labels':        all_labels,
        'groups':        all_groups,
        # Axis and window metadata
        'xaxis':         xaxis_notched,
        'n_bands':       n_fit_bands,
        'wn_range':      wn_range,
        'spectral_range': (
            f"Channels {ch_start}-{ch_end - 1} "
            f"({xaxis_fit[0]:.1f}–{xaxis_fit[-1]:.1f} cm⁻¹)"
        ),
        'spectral_channel_low':  ch_start,
        'spectral_channel_high': ch_end - 1,
        # Endmember libraries (for reference / sum_group_conc)
        'endlib':       endlib,
        'forcedlib':    forcedlib,
        'atmlib':       atmlib,
        'E_fit':        E_fit_out,      # (n_out_lib + bb_int + slope_int, n_fit_bands)
        'measured_fit': out_measured_fit,
        # Metadata
        'has_slope':         bool(slope),
        'slope_t_center':    _slope_t_center if slope else None,
        'slope_seed_dt':     slope_seed_dt   if slope else None,
        'delta_t_estimated': out_delta_t,
        'algorithm':    algo,
        'notchco2':     notchco2_info,
        'excluded':     excluded_info,
        'sample_labels': sample_labels,
        'covm':         last_covm,
        'nsamples':     n_fit_bands,    # alias used by sum_group_conc
    }

    # Optional outputs
    if out_err is not None:
        smaout['error']      = out_err
        smaout['bberror']    = out_bberr
        smaout['slopeerror'] = out_slopeerr
        smaout['normerror']  = out_normerr

    if out_sort is not None:
        smaout['sort'] = out_sort

    if out_modsur is not None:
        smaout['modsur'] = out_modsur
        smaout['rematm'] = out_rematm

    if out_atm_conc is not None:
        smaout['atm_conc']   = out_atm_conc
        smaout['atm_labels'] = atm_labels

    # ------------------------------------------------------------------
    # 19. Group concentrations
    # Ref: spectral_tools.dvrc::sma, lines 5414-5453
    # ------------------------------------------------------------------
    if group:
        if not any(g not in ('Unknown', '') for g in all_groups):
            logging.warning(
                "group=True but no 'category' field found in library; "
                "skipping group summation."
            )
        else:
            smaout['grouped'] = sum_group_conc(smaout)

            # Extract and separate 'Black body' from grouped output
            gp = smaout['grouped']
            pos_bb = gp['grouped_labels'].index('Black body')

            gp['grouped_bb'] = gp['grouped_conc'][..., pos_bb]
            gp['grouped_conc']   = np.delete(gp['grouped_conc'],   pos_bb, axis=-1)
            gp['grouped_labels'] = [
                l for i, l in enumerate(gp['grouped_labels']) if i != pos_bb
            ]

            if 'grouped_error' in gp:
                gp['grouped_bberror'] = gp['grouped_error'][..., pos_bb]
                gp['grouped_error']   = np.delete(gp['grouped_error'], pos_bb, axis=-1)

            # Extract and separate 'Slope' from grouped output (if present)
            if slope and 'Slope' in gp['grouped_labels']:
                pos_sl = gp['grouped_labels'].index('Slope')
                gp['grouped_slope']  = gp['grouped_conc'][..., pos_sl]
                gp['grouped_conc']   = np.delete(gp['grouped_conc'],   pos_sl, axis=-1)
                gp['grouped_labels'] = [
                    l for i, l in enumerate(gp['grouped_labels']) if i != pos_sl
                ]
                if 'grouped_error' in gp:
                    gp['grouped_slopeerror'] = gp['grouped_error'][..., pos_sl]
                    gp['grouped_error']      = np.delete(gp['grouped_error'], pos_sl, axis=-1)

            # grouped_normconc denominator = mineral groups only (BB and slope excluded)
            gp_sum = np.sum(gp['grouped_conc'], axis=-1, keepdims=True)
            gp['grouped_normconc'] = np.where(
                gp_sum > 0, gp['grouped_conc'] / gp_sum * 100.0, 0.0
            )
            if 'grouped_error' in gp:
                gp['grouped_normerror'] = np.where(
                    gp_sum > 0, gp['grouped_error'] / gp_sum * 100.0, 0.0
                )

            gp['grouped_sort'] = sort_cube(gp['grouped_conc'])

    logging.info(
        "SMA complete. Mean RMS: %.4f  |  Mean BB: %.3f",
        float(np.nanmean(smaout['rms'])),
        float(np.nanmean(smaout['bb'])),
    )

    if plot or save_plots:
        plot_sma(smaout, save_plots=save_plots,
                 group=plot_group, residual=plot_residual, error=plot_error,
                 cumulative=plot_cumulative, other=plot_other, offset=plot_offset)

    if save:
        out_dir   = os.path.abspath(save_path) if save_path else os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base      = os.path.join(out_dir, f"sma_results_{timestamp}")
        utils.saveHDF(smaout, base + ".hdf")
        logging.info("Saved SMA HDF → %s.hdf", base)
        utils.save_sma_csv(smaout, base + ".csv", group=group)

    return smaout


def summary_sma(
    out: dict,
    sample: str | int | None = None,
    group: bool = False,
    threshold: float = 0.01,
) -> None:
    """
    Pretty-print a concentration summary table for SMA results.

    Mirrors ``spectral_tools.dvrc::summary_sma`` (lines 7387–7671).  For each
    requested sample the table lists endmembers sorted by normalised
    concentration, along with raw abundance, BB fraction, and RMS error.
    Optional ± errors are shown when ``calc_errors=True`` was used in
    :func:`sma`.

    Parameters
    ----------
    out : dict
        Output dict from :func:`sma`.
    sample : str, int, or None
        Which sample(s) to show.  A string is matched against
        ``out['sample_labels']``; an int is treated as a 0-based index.
        ``None`` (default) prints all samples.
    group : bool
        If True, use the grouped concentrations from ``out['grouped']``
        instead of the per-endmember concentrations.  Silently falls back
        to per-endmember if ``'grouped'`` is absent.
    threshold : float
        Minimum normalised concentration (%) below which an endmember is
        omitted from the table.  Default 0.01.
    """
    sample_labels = out.get('sample_labels', [])
    n_samples     = len(sample_labels)

    # ── Resolve which samples to print ──────────────────────────────────────
    if sample is None:
        indices = list(range(n_samples))
    elif isinstance(sample, int):
        indices = [sample]
    else:
        if sample not in sample_labels:
            raise ValueError(f"Sample '{sample}' not found in results.")
        indices = [sample_labels.index(sample)]

    # ── Choose grouped vs per-endmember arrays ───────────────────────────────
    gp = out.get('grouped') if group else None
    if group and gp is None:
        logging.warning("summary_sma: group=True but 'grouped' key absent — using per-endmember.")

    if gp is not None:
        labels        = gp['grouped_labels']
        conc_arr      = np.asarray(gp['grouped_conc'])
        normconc_arr  = np.asarray(gp['grouped_normconc'])
        bb_arr        = np.asarray(gp.get('grouped_bb',    np.zeros(n_samples)))
        slope_arr     = np.asarray(gp.get('grouped_slope', np.zeros(n_samples)))
        error_arr     = np.asarray(gp['grouped_error'])      if 'grouped_error'      in gp else None
        bberror_arr   = np.asarray(gp['grouped_bberror'])    if 'grouped_bberror'    in gp else None
        slopeerror_arr= np.asarray(gp['grouped_slopeerror']) if 'grouped_slopeerror' in gp else None
    else:
        labels        = out.get('labels', [])
        conc_arr      = np.asarray(out['conc'])
        normconc_arr  = np.asarray(out['normconc'])
        bb_arr        = np.asarray(out['bb'])
        slope_arr     = np.asarray(out.get('slope', np.zeros(n_samples)))
        error_arr     = np.asarray(out['error'])      if 'error'      in out else None
        bberror_arr   = np.asarray(out['bberror'])    if 'bberror'    in out else None
        slopeerror_arr= np.asarray(out['slopeerror']) if 'slopeerror' in out else None

    has_slope = np.any(slope_arr != 0)

    rms_arr     = np.asarray(out['rms'])
    has_errors  = error_arr is not None

    # ── Column widths ────────────────────────────────────────────────────────
    SEP  = '─' * 75
    col0 = 28   # endmember name

    # ── Header template ──────────────────────────────────────────────────────
    if has_errors:
        hdr = (f"{'Endmember':<{col0}}  {'Abundance (%)':>16}  "
               f"{'Norm. (%)':>14}")
    else:
        hdr = (f"{'Endmember':<{col0}}  {'Abundance (%)':>14}  "
               f"{'Norm. (%)':>12}")

    for idx in indices:
        lbl       = sample_labels[idx] if idx < n_samples else f'Sample {idx}'
        conc_i    = conc_arr[idx]
        norm_i    = normconc_arr[idx]
        bb_i      = float(bb_arr[idx])
        slope_i   = float(slope_arr[idx])
        rms_i     = float(rms_arr[idx])
        err_i     = error_arr[idx]      if has_errors else None
        bberr_i   = bberror_arr[idx]    if has_errors else None
        slopeerr_i= slopeerror_arr[idx] if (has_errors and slopeerror_arr is not None) else None

        print(f"\n{'━' * 75}")
        print(f"  Summary — {lbl}")
        print(f"{'━' * 75}")
        print(hdr)
        print(SEP)

        # Sort by normconc descending
        order = np.argsort(norm_i)[::-1]
        for j in order:
            nc = float(norm_i[j])
            if nc < threshold:
                continue
            name = str(labels[j])[:col0]
            ab   = float(conc_i[j]) * 100.0
            if has_errors:
                er  = float(err_i[j]) * 100.0
                total = float(conc_i.sum()) or 1.0
                ner = float(err_i[j]) / total * 100.0
                print(f"  {name:<{col0}}  {ab:>9.2f} ± {er:>5.2f}  {nc:>9.2f} ± {ner:>5.2f}")
            else:
                print(f"  {name:<{col0}}  {ab:>12.2f}  {nc:>12.2f}")

        print(SEP)

        total_sum = float(conc_i.sum() + bb_i + slope_i) * 100.0
        norm_sum  = float(norm_i.sum())
        print(f"  {'Sum (BB + slope included)':<{col0}}  {total_sum:>12.2f}")
        if norm_sum > 0.0:
            print(f"  {'Sum (normalised)':<{col0}}  {norm_sum:>12.2f}")
        if has_errors:
            bberr_val   = float(bberr_i)    * 100.0 if bberr_i   is not None else 0.0
            print(f"  {'Blackbody abundance':<{col0}}  {bb_i * 100:>9.2f} ± {bberr_val:>5.2f}")
        else:
            print(f"  {'Blackbody abundance':<{col0}}  {bb_i * 100:>12.2f}")
        if has_slope:
            if has_errors and slopeerr_i is not None:
                slerr_val = float(slopeerr_i) * 100.0
                print(f"  {'Slope abundance':<{col0}}  {slope_i * 100:>9.2f} ± {slerr_val:>5.2f}")
            else:
                print(f"  {'Slope abundance':<{col0}}  {slope_i * 100:>12.2f}")
        print(f"  {'RMS error':<{col0}}  {rms_i:>12.5f}")
        print()


# =============================================================================
# ========================= emissivity functions ==============================
# =============================================================================

# Sentinel used to distinguish "not provided" from any valid value (including
# None) in emissivity_nem().  Enables instrument presets to set defaults while
# still letting callers override individual parameters explicitly.
_UNSET = object()

# Shared lab-spectrometer defaults (ASU lab1/lab2/lab3, NAU, generic).
# Ref: spectral_tools.dvrc::emissivity(), lab branch, lines ~600–650
_LAB_PRESET: dict = dict(
    wn_range         = (500.0, 1700.0),
    wn_range_cold    = (450.0,  900.0),
    threshold_t_cold = 160.0,
    threshold_t_warm = 170.0,
    filter_size      = 3,
    max_emiss        = 1.0,
    max_emiss_low    = None,   # None → falls back to max_emiss
    co2_range        = None,   # None → no exclusion
)

INSTRUMENT_PRESETS: dict[str, dict] = {
    # --- Laboratory FTIR spectrometers ---
    'spectrometer': _LAB_PRESET,
    'nau':          _LAB_PRESET,
    'asu':          _LAB_PRESET,
    'swri':         _LAB_PRESET,
    # --- TES (Mars Global Surveyor Thermal Emission Spectrometer) ---
    # CO2 range per TES Data Processing User's Guide (500–800 cm⁻¹).
    # max_emiss_low = 0.97 for cold-target branch per TES processing.
    # Ref: spectral_tools.dvrc::emissivity(), tes branch, ~lines 660–710
    'tes': dict(
        wn_range         = (300.0, 1350.0),
        wn_range_cold    = (300.0,  500.0),
        threshold_t_cold = 215.0,
        threshold_t_warm = 225.0,
        filter_size      = 7,
        max_emiss        = 1.0,
        max_emiss_low    = 0.97,
        co2_range        = (500.0, 800.0),
    ),
    # TES5: same spectral range as TES, wider CO2 exclusion window.
    # Ref: spectral_tools.dvrc::emissivity(), tes5 branch, co2lowchan=co2chan-31
    'tes5': dict(
        wn_range         = (300.0, 1350.0),
        wn_range_cold    = (300.0,  500.0),
        threshold_t_cold = 215.0,
        threshold_t_warm = 225.0,
        filter_size      = 7,
        max_emiss        = 1.0,
        max_emiss_low    = 0.97,
        co2_range        = (574.0, 742.0),
    ),
    # --- Mini-TES (Mars Exploration Rover) ---
    # CO2 range: ±10 channels at ~9.99 cm⁻¹/ch centred on 669 cm⁻¹.
    # Ref: spectral_tools.dvrc::emissivity(), mtes branch, ~lines 730–770
    'mtes': dict(
        wn_range         = (500.0, 1400.0),
        wn_range_cold    = (450.0,  900.0),
        threshold_t_cold = 220.0,
        threshold_t_warm = 230.0,
        filter_size      = 7,
        max_emiss        = 1.0,
        max_emiss_low    = None,
        co2_range        = (569.0, 769.0),
    ),
}

# Keys in INSTRUMENT_PRESETS that correspond to laboratory FTIR spectrometers.
# Used by emcal() to select emissivity_hullfit (full, n_bb=3) vs
# emissivity_hullfit_linear (fast, n_bb=2) when method='hullfit'.
_LAB_INSTRUMENTS: frozenset[str] = frozenset({'spectrometer', 'nau', 'asu', 'swri'})



def emissivity_nem(
    wn: np.ndarray,
    data: np.ndarray,
    inst: str | None = None,
    max_emiss: float = _UNSET,
    max_emiss_low: float | None = _UNSET,
    wn_range: tuple[float, float] = _UNSET,
    wn_range_cold: tuple[float, float] = _UNSET,
    filter_size: int = _UNSET,
    threshold_t_cold: float = _UNSET,
    threshold_t_warm: float = _UNSET,
    co2_range: tuple[float, float] | None = _UNSET,
    toffset: float = 0.0,
    downwelling_t: float = 0.0,
    downwelling_e: float = 1.0,
    downwelling_rad: np.ndarray | None = None,
) -> dict:
    """
    Retrieve emissivity via the Normalized Emissivity Method (NEM) with
    optional downwelling radiance correction.

    Direct translation of DaVinci's ``emissivity()`` (spectral_tools.dvrc,
    lab-spectrometer branch, lines 600–979, 1111–1119).

    Two-stage brightness-temperature detection: the full wavenumber range
    is used to find warm-target temperatures; a narrower cold range handles
    cold scenes; a hysteresis threshold makes the transition smooth.  An
    optional per-sample downwelling correction is applied before dividing
    by the ideal blackbody curve.

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹), shape (n_bands,).
    data : np.ndarray
        Calibrated sample radiance (mW m⁻² sr⁻¹ cm), shape (n_bands,).
    inst : str or None
        Instrument preset name.  Loads default values for ``wn_range``,
        ``wn_range_cold``, ``filter_size``, ``threshold_t_cold``,
        ``threshold_t_warm``, ``max_emiss``, ``max_emiss_low``, and
        ``co2_range`` from ``INSTRUMENT_PRESETS[inst]``.  Any of those
        parameters explicitly passed as keyword arguments override the
        preset.  Valid keys: ``'lab'``, ``'nau'``, ``'tes'``, ``'tes5'``,
        ``'mtes'``.  ``None`` falls back to the lab defaults.
    max_emiss : float
        Maximum emissivity ceiling used when computing warm-target brightness
        temperature (T1 stage).  Default 1.0.
    max_emiss_low : float or None
        Separate maximum emissivity ceiling for the cold-target brightness
        temperature (T2 stage).  If None, falls back to ``max_emiss``.
        Ref: spectral_tools.dvrc::emissivity, TES processing sets this to
        0.97 for the cold branch while keeping ``max_emiss`` = 1.0.
    wn_range : tuple[float, float]
        ``(wn_low, wn_high)`` window (cm⁻¹) for the warm-target brightness
        temperature search.  Default ``(500, 1700)``.
    wn_range_cold : tuple[float, float]
        ``(wn_low, wn_high)`` narrower window (cm⁻¹) for the cold-target
        brightness temperature search.  Default ``(450, 900)``.
    filter_size : int
        Box-car kernel width applied to brightness temperature before peak
        detection.  Default 3.
    threshold_t_cold : float
        If the cold-range peak brightness temperature is below this value (K)
        the target is classified as cold.  Default 160 K.
    threshold_t_warm : float
        If the full-range peak brightness temperature is above this value (K)
        the target is classified as warm.  Default 170 K.
    co2_range : tuple[float, float] or None
        ``(wn_low, wn_high)`` interval (cm⁻¹) to zero out before searching
        for maximum brightness temperature (both T1 and T2 stages).  Useful
        for suppressing the CO₂ absorption band (~667 cm⁻¹) when present.
        Ref: spectral_tools.dvrc::emissivity, co2lowchan/co2highchan logic.
    toffset : float
        Additive offset (K) applied to the derived target temperature before
        computing the ideal blackbody radiance.  Corrects for known systematic
        temperature biases.  Default 0.0 (no offset).
        Ref: spectral_tools.dvrc::emissivity, toffset parameter.
    downwelling_t : float
        Temperature (K) of the downwelling radiance source (e.g. the
        environmental chamber).  Set to 0 to skip correction (default).
    downwelling_e : float
        Emissivity of the downwelling radiance source.  Default 1.0.
    downwelling_rad : np.ndarray or None
        Pre-computed downwelling radiance spectrum.  When provided it takes
        precedence over ``downwelling_t`` / ``downwelling_e``.

    Returns
    -------
    dict
        Keys: ``wn``, ``data``, ``rad_bb``, ``emiss``, ``temp``,
        ``wn_t1``, ``wn_t2``, ``max_t1``, ``max_t2``,
        ``tb``, ``tb_smooth1``, ``tb_smooth2``,
        ``inst``, ``wn_range``, ``wn_range_cold``, ``filter_size``,
        ``max_emiss``, ``max_emiss_low``, ``co2_range``, ``toffset``,
        ``threshold_t_cold``, ``threshold_t_warm``,
        ``downwelling_t``, ``downwelling_e``, ``downwelling_rad``.
    """
    # --- Instrument preset resolution ---
    if inst is not None and inst not in INSTRUMENT_PRESETS:
        raise ValueError(
            f"Unknown instrument preset '{inst}'. "
            f"Valid options: {list(INSTRUMENT_PRESETS)}"
        )
    preset = INSTRUMENT_PRESETS[inst] if inst is not None else _LAB_PRESET
    if max_emiss      is _UNSET: max_emiss      = preset['max_emiss']
    if max_emiss_low  is _UNSET: max_emiss_low  = preset['max_emiss_low']
    if wn_range       is _UNSET: wn_range       = preset['wn_range']
    if wn_range_cold  is _UNSET: wn_range_cold  = preset['wn_range_cold']
    if filter_size    is _UNSET: filter_size     = preset['filter_size']
    if threshold_t_cold is _UNSET: threshold_t_cold = preset['threshold_t_cold']
    if threshold_t_warm is _UNSET: threshold_t_warm = preset['threshold_t_warm']
    if co2_range      is _UNSET: co2_range      = preset['co2_range']

    if inst is not None:
        logging.info("emissivity_nem: using '%s' instrument preset", inst)

    # Ref: spectral_tools.dvrc::emissivity, lines 820–958
    if max_emiss_low is None:
        max_emiss_low = max_emiss

    wave1, wave2     = wn_range
    w1_cold, w2_cold = wn_range_cold
    kernel = np.ones(filter_size) / filter_size

    # --- Step 1: brightness temperature ---
    # Pre-compute downwelling radiance to correct the BBT input: without this,
    # the downwelling contribution in `data` inflates T_nem, causing max(emiss) < max_emiss.
    # Forward model at peak: data = max_emiss*BB(T) + (1-max_emiss)*dw_rad
    # → BB(T) = (data - (1-max_emiss)*dw_rad) / max_emiss
    if downwelling_rad is not None:
        _dw_rad = downwelling_rad
    elif downwelling_t != 0.0:
        _dw_rad = downwelling_e * utils.rad(wn, downwelling_t)
    else:
        _dw_rad = None

    if _dw_rad is not None:
        data_bt = data - (1.0 - max_emiss) * _dw_rad
    else:
        data_bt = data
    tb = utils.bbt(wn, data_bt / max_emiss)

    # --- Step 2a: warm-target brightness temperature (full wn range) ---
    t1 = tb.copy()
    t1[(wn < wave1) | (wn > wave2)] = 0.0
    if co2_range is not None:
        t1[(wn >= co2_range[0]) & (wn <= co2_range[1])] = 0.0
    tb_smooth1 = np.convolve(t1, kernel, mode='same')
    max_t1     = float(tb_smooth1.max())
    wn_t1      = float(wn[np.argmax(tb_smooth1)])

    # --- Step 2b: cold-target brightness temperature (narrow range) ---
    # Ref: spectral_tools.dvrc::emissivity, max_emiss_low branch
    if max_emiss_low != max_emiss:
        if _dw_rad is not None:
            data_bt_low = data - (1.0 - max_emiss_low) * _dw_rad
        else:
            data_bt_low = data
        tb2 = utils.bbt(wn, data_bt_low / max_emiss_low)
    else:
        tb2 = tb
    t2  = tb2.copy()
    t2[(wn < w1_cold) | (wn > w2_cold)] = 0.0
    if co2_range is not None:
        t2[(wn >= co2_range[0]) & (wn <= co2_range[1])] = 0.0
    tb_smooth2 = np.convolve(t2, kernel, mode='same')
    max_t2     = float(tb_smooth2.max())
    wn_t2      = float(wn[np.argmax(tb_smooth2)])

    # --- Step 3: select target temperature with hysteresis ---
    if max_t1 > threshold_t_warm:
        temp = max_t1
    elif max_t2 < threshold_t_cold:
        temp = max_t2
    else:
        temp = (max_t1 + max_t2) / 2.0

    # --- Step 3b: optional temperature offset ---
    # Ref: spectral_tools.dvrc::emissivity, toffset parameter
    if toffset != 0.0:
        logging.info("Applying temperature offset of %.2f K", toffset)
        temp += toffset

    logging.info(
        "Sample temperature: %.1f K  (max_t1=%.1f @ %.0f cm⁻¹, max_t2=%.1f @ %.0f cm⁻¹)",
        temp, max_t1, wn_t1, max_t2, wn_t2,
    )

    # --- Step 4: ideal blackbody radiance at target temperature ---
    rad_bb = utils.rad(wn, temp)

    # --- Step 5: emissivity with optional downwelling correction ---
    # Ref: spectral_tools.dvrc::emissivity, lines 932–947
    if downwelling_t == 0.0 and downwelling_rad is None:
        emiss = data / rad_bb
    else:
        if downwelling_rad is None:
            downwelling_rad = downwelling_e * utils.rad(wn, downwelling_t)
        emiss = (data - downwelling_rad) / (rad_bb - downwelling_rad)

    # Zero out channels with non-positive measured radiance
    emiss[data <= 0.0] = 0.0

    return {
        'wn':               wn,
        'data':             data,
        'rad_bb':           rad_bb,
        'emiss':            emiss,
        'temp':             temp,
        'wn_t1':            wn_t1,
        'wn_t2':            wn_t2,
        'max_t1':           max_t1,
        'max_t2':           max_t2,
        'tb':               tb,
        'tb_smooth1':       tb_smooth1,
        'tb_smooth2':       tb_smooth2,
        'inst':             inst,
        'wn_range':         wn_range,
        'wn_range_cold':    wn_range_cold,
        'filter_size':      filter_size,
        'max_emiss':        max_emiss,
        'max_emiss_low':    max_emiss_low,
        'co2_range':        co2_range,
        'toffset':          toffset,
        'threshold_t_cold': threshold_t_cold,
        'threshold_t_warm': threshold_t_warm,
        'downwelling_t':    downwelling_t,
        'downwelling_e':    downwelling_e,
        'downwelling_rad':  downwelling_rad,
    }


def emissivity_mmd(
    wn: np.ndarray,
    data: np.ndarray,
    max_emiss: float = 1.0,
) -> dict:
    """
    Retrieve emissivity using the Min-Max Difference (MMD) algorithm.

    Iteratively refines the emissivity spectrum by anchoring the minimum
    emissivity to an empirical relationship with the spectral min-max
    difference until convergence (Salisbury & D'Aria 1992).

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹).
    data : np.ndarray
        Calibrated sample radiance, same length as *wn*.
    max_emiss : float
        Initial maximum emissivity normalisation value.

    Returns
    -------
    dict
        Keys: ``wn``, ``data``, ``rad0``, ``emiss``, ``nfev``, ``r2``,
        ``temp``.

    Raises
    ------
    RuntimeError
        If the iterative refinement does not converge within 10 iterations.
    """
    idx = (wn > 850) & (wn < 1250)

    temp0 = utils.bbt(wn, data / max_emiss).max()

    emiss0 = (data / utils.rad(wn, temp0)) / (data.mean() / utils.rad(wn, temp0).mean())

    converge = False
    max_iter_reached = False
    loop = 0

    while not converge and not max_iter_reached:

        loop += 1

        mmd = emiss0[idx].max() - emiss0[idx].min()
        emin = 0.994 - 0.687 * mmd ** 0.737
        emiss = emiss0 * (emin / emiss0.min())
        r2 = r2_score(emiss, emiss0)
        logging.debug("iter %d  r²=%.6f", loop, r2)

        if r2 >= 0.9999:
            converge = True
        else:
            emiss0 = emiss

        if loop > 10:
            max_iter_reached = True

    if converge:

        emiss = emiss / max_emiss
        rad0 = data / emiss
        temp = utils.bbt(wn, rad0).mean()

        out = {}
        out["wn"] = wn
        out["data"] = data
        out["rad0"] = rad0
        out["emiss"]  = emiss
        out["nfev"] = loop
        out["r2"] = r2
        out["temp"] = temp

        return out
    else:
        raise RuntimeError("Did not converge")


def emissivity_alpha(
    wn: np.ndarray,
    data: np.ndarray,
    wn_range: tuple[float, float] | None = (500.0, 1700.0),
    co2_range: tuple[float, float] | None = None,
    max_emiss: float = 1.0,
    downwelling_t: float = 0.0,
    downwelling_e: float = 1.0,
    downwelling_rad: np.ndarray | None = None,
) -> dict:
    """
    Retrieve emissivity via the Alpha Residuals method with optional
    downwelling correction.

    Emissivity spectral shape is determined by dividing the net surface
    emission at each channel by a single reference Planck curve at T_ref,
    then rescaling so the maximum over the fitting window equals max_emiss.
    T_ref is the mean brightness temperature across the fitting window (more
    robust than the single peak-band temperature used by NEM), so the
    provisional emissivity is uniformly below unity; the rescaling step
    restores the correct absolute level while preserving spectral shape.

    Algorithm
    ---------
    1. Net surface emission::

           data_net = data - (1 - max_emiss) * dw_rad

       Uses max_emiss as an initial emissivity estimate to remove the
       reflected downwelling component.

    2. Reference temperature: mean BBT over the fit window assuming
       max_emiss surface emission::

           T_ref = nanmean_j( BBT( data_net_j / max_emiss ) )

    3. Provisional emissivity at T_ref, rescaled so the fit-window
       maximum equals max_emiss::

           emiss_prov_i = data_net_i / B_i(T_ref)
           emiss_i      = max_emiss * emiss_prov_i / max_j( emiss_prov_j )

    4. Final temperature: median BBT of the full radiometric inversion
       over the fit window::

           surf_em_j = ( data_j - (1 - emiss_j) * dw_j ) / emiss_j
           temp      = nanmedian_j( BBT( surf_em_j ) )

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹), shape (n_bands,).
    data : np.ndarray
        Calibrated sample radiance, same length as wn.
    wn_range : tuple[float, float] or None
        Fitting wavenumber window (wn_min, wn_max) cm⁻¹ used for the
        T_ref mean and the emissivity rescaling maximum.  None uses the
        full axis.  Default ``(500, 1700)``.
    co2_range : tuple[float, float] or None
        Wavenumber interval to exclude (CO₂ band), cm⁻¹.
    max_emiss : float
        Maximum emissivity constraint used to (a) subtract reflected
        downwelling and (b) set the emissivity ceiling.  Default 1.0.
    downwelling_t : float
        Temperature (K) of the downwelling radiance source.  Set to 0
        to skip correction (default).
    downwelling_e : float
        Emissivity of the downwelling source.  Default 1.0.
    downwelling_rad : np.ndarray or None
        Pre-computed downwelling radiance spectrum.  Takes precedence
        over downwelling_t / downwelling_e when provided.

    Returns
    -------
    dict
        Keys: ``wn``, ``data``, ``data0_eff``, ``data_net``, ``emiss``,
        ``temp``, ``t_ref``, ``rad_bb``, ``wn_range``, ``co2_range``,
        ``max_emiss``, ``downwelling_t``, ``downwelling_e``,
        ``downwelling_rad``, ``elapsed``.
    """
    start = time.perf_counter()

    # --- Downwelling radiance ---
    if downwelling_rad is not None:
        _dw_rad = np.asarray(downwelling_rad, dtype=float)
    elif downwelling_t != 0.0:
        _dw_rad = downwelling_e * utils.rad(wn, downwelling_t)
    else:
        _dw_rad = np.zeros(len(wn))

    # --- Step 1: net surface emission ---
    data_net = data - (1.0 - max_emiss) * _dw_rad

    # --- Fit-window mask ---
    if wn_range is not None:
        wn1, wn2 = wn_range
        fit_mask = (wn >= wn1) & (wn <= wn2)
    else:
        fit_mask = np.ones(len(wn), dtype=bool)
    if co2_range is not None:
        fit_mask &= (wn < co2_range[0]) | (wn > co2_range[1])

    wn_fit       = wn[fit_mask]
    data_net_fit = data_net[fit_mask]

    # --- Step 2: reference temperature — mean BBT over fit window ---
    bbt_init = utils.bbt(
        wn_fit,
        np.where(data_net_fit > 0, data_net_fit / max_emiss, np.nan),
    )
    t_ref = float(np.nanmean(bbt_init))

    logging.info("emissivity_alpha: T_ref = %.1f K (mean BBT over fit window)", t_ref)

    # --- Step 3: provisional emissivity, rescaled so fit-window max = max_emiss ---
    planck_ref = utils.rad(wn, t_ref)
    emiss_prov = np.where(data_net > 0, data_net / planck_ref, np.nan)
    emiss_max  = float(np.nanmax(emiss_prov[fit_mask]))
    emiss      = max_emiss * emiss_prov / emiss_max
    emiss[~np.isfinite(emiss) | (data_net <= 0)] = 0.0

    # --- Step 4: final temperature — median BBT of full radiometric inversion ---
    e_fit  = emiss[fit_mask]
    d_fit  = data[fit_mask]
    dw_fit = _dw_rad[fit_mask]
    valid  = e_fit > 0
    surf_em        = np.full(int(fit_mask.sum()), np.nan)
    surf_em[valid] = (d_fit[valid] - (1.0 - e_fit[valid]) * dw_fit[valid]) / e_fit[valid]
    surf_em        = np.where(surf_em > 0, surf_em, np.nan)
    bbt_final      = utils.bbt(wn_fit, surf_em)
    temp           = float(np.nanmedian(bbt_final))

    logging.info("emissivity_alpha: temp = %.1f K (median BBT inversion)", temp)

    rad_bb    = utils.rad(wn, t_ref)
    data0_eff = (data - _dw_rad) / max_emiss

    stop = time.perf_counter()

    return {
        'wn':              wn,
        'data':            data,
        'data0_eff':       data0_eff,
        'data_net':        data_net,
        'emiss':           emiss,
        'temp':            temp,
        't_ref':           t_ref,
        'rad_bb':          rad_bb,
        'wn_range':        wn_range,
        'co2_range':       co2_range,
        'max_emiss':       max_emiss,
        'downwelling_t':   downwelling_t,
        'downwelling_e':   downwelling_e,
        'downwelling_rad': _dw_rad,
        'elapsed':         stop - start,
    }


def _upper_hull_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask marking points on the upper convex hull of (x, y).

    Parameters
    ----------
    x, y : np.ndarray
        1-D coordinate arrays of equal length.

    Returns
    -------
    np.ndarray of bool
        True at indices that lie on the upper convex hull.
    """
    order = np.argsort(x)

    stack: list[int] = []
    for i in order:
        while len(stack) >= 2:
            o, a = stack[-2], stack[-1]
            cross = (x[a] - x[o]) * (y[i] - y[o]) - (y[a] - y[o]) * (x[i] - x[o])
            if cross >= 0:   # a is at or below line o->i -- remove from upper hull
                stack.pop()
            else:
                break
        stack.append(i)

    mask = np.zeros(len(x), dtype=bool)
    mask[stack] = True
    return mask


def emissivity_hullfit(
    wn: np.ndarray,
    data: np.ndarray,
    n_bb: int = 2,
    temp_range: list[float] | None = None,
    temp_halfwidth: float = 50.0,
    temp_step: float = 5.0,
    wn_range: tuple[float, float] | None = (500.0, 1700.0),
    co2_range: tuple[float, float] | None = None,
    max_emiss: float = 1.0,
    max_iter: int = 30,
    violation_weight: float = 5.0,
    violation_tol: float = 0.0,
    escalation_factor: float = 4.0,
    max_escalations: int = 4,
    downwelling_t: float = 0.0,
    downwelling_e: float = 1.0,
    downwelling_rad: np.ndarray | None = None,
    plotout: bool = False,
    reverse: bool = False,
) -> dict:
    """
    Retrieve emissivity via convex-hull fitting of a Planck mixture with strict
    upper-bound enforcement.

    Two-stage algorithm:

    **Stage 1 — NNLS on a temperature grid.**  Builds a Planck matrix whose
    columns are blackbody curves at uniformly spaced temperatures, solves a
    non-negative least-squares problem to find the temperature distribution,
    and seeds the *n_bb* parametric components from the dominant weight peaks.

    **Stage 2 — Convex-hull seeding + violation repair.**  The upper convex
    hull of ``(wn_fit, data_fit)`` is computed geometrically and used as the
    initial fitting set.  After each parametric fit the model is evaluated over
    the full fitting range; any channel where ``data > model`` is a violation of
    the upper-bound constraint and is added to the fitting set with weight
    ``violation_weight``.  Iteration continues until no violations remain or
    *max_iter* is reached.  The strict enforcement guarantees
    ``model >= data`` at convergence.

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹).
    data : np.ndarray
        Calibrated sample radiance, same length as *wn*.
    n_bb : int
        Number of blackbody components in the Planck mixture.
    temp_range : list[float] or None
        Explicit ``[T_min, T_max]`` bounds (K) for the NNLS grid and the
        parametric fit.  When ``None`` (default), the range is derived
        automatically as ``[T_peak − temp_halfwidth, T_peak + temp_halfwidth]``
        where *T_peak* is the peak brightness temperature of the measured
        radiance within the fitting window.
    temp_halfwidth : float
        Half-width (K) of the automatically derived temperature range.
        Ignored when *temp_range* is supplied explicitly.
    temp_step : float
        Temperature grid spacing (K) for the NNLS stage.
    wn_range : tuple[float, float] or None
        Fitting wavenumber window ``(wn_min, wn_max)`` cm⁻¹.  ``None`` uses
        the full axis.
    co2_range : tuple[float, float] or None
        Wavenumber interval to exclude from fitting (CO₂ band), cm⁻¹.
    max_emiss : float
        Maximum emissivity normalisation applied before fitting.
    max_iter : int
        Maximum number of violation-repair iterations.
    violation_weight : float
        Relative weight applied to channels where ``data > model``; they are
        fitted with ``sigma = 1 / violation_weight`` to force the model up.
    violation_tol : float
        Fractional slack on the violation criterion.  A channel is a violation
        only when ``data > model * (1 + violation_tol)``.  The default 0.0
        enforces strict ``model >= data``; a small positive value (e.g. 0.01)
        allows 1 % over-shoot before a channel is treated as a violation.
    escalation_factor : float
        When the hull set stops growing (optimizer stuck with residual
        violations), multiply the effective violation weight by this factor and
        retry.  Applied up to *max_escalations* times before giving up.
    max_escalations : int
        Maximum number of weight-escalation attempts after the hull set stalls.
    downwelling_t : float
        Downwelling air temperature (K).  0.0 disables correction.
    downwelling_e : float
        Downwelling emissivity fraction.
    downwelling_rad : np.ndarray or None
        Pre-computed downwelling radiance array.  Takes precedence over the
        scalar ``downwelling_t`` / ``downwelling_e`` pair.
    plotout : bool
        If True, display a diagnostic figure (radiance + model overlay and
        emissivity panels).
    reverse : bool
        If True, reverse the wavenumber axis in diagnostic plots.

    Returns
    -------
    dict
        Keys: ``wn0``, ``data0``, ``data0_eff``, ``wn``, ``data``, ``model``,
        ``model_err``, ``res``, ``model0``, ``model0_err``, ``res0``,
        ``emiss``, ``emiss_err``, ``fwd``, ``n_bb``, ``p0``, ``bounds``,
        ``temp_range``, ``wn_range``, ``co2_range``, ``temp_step``,
        ``violation_weight``, ``violation_tol``, ``max_emiss``, ``popt``, ``pcov``, ``perr``,
        ``nerr``, ``nfev``, ``r2``, ``r20``, ``rmse``, ``rmse0``,
        ``bb_temps``, ``bb_temps_err``, ``bb_fracs``, ``bb_fracs_err``,
        ``temp``, ``rad_bb``, ``temp_grid``, ``nnls_weights``,
        ``n_violations_final``, ``elapsed``.
    """
    start = time.perf_counter()

    wn0   = wn.copy()
    data0 = data.copy() / max_emiss

    if downwelling_t == 0.0 and downwelling_rad is None:
        dw_rad = np.zeros(len(wn0))
    else:
        if downwelling_rad is None:
            downwelling_rad = downwelling_e * utils.rad(wn0, downwelling_t)
        dw_rad = downwelling_rad

    data0_eff = (data - dw_rad) / max_emiss

    if wn_range is not None:
        wn1, wn2 = wn_range
        fit_mask = (wn0 >= wn1) & (wn0 <= wn2)
    else:
        fit_mask = np.ones(len(wn0), dtype=bool)
    if co2_range is not None:
        fit_mask &= (wn0 < co2_range[0]) | (wn0 > co2_range[1])
    wn_fit   = wn0[fit_mask]
    data_fit = data0_eff[fit_mask]

    # Auto-derive temperature range from peak brightness temperature.
    if temp_range is None:
        bt_fit  = utils.bbt(wn_fit, data_fit)
        t_peak  = float(bt_fit[np.isfinite(bt_fit)].max())
        temp1   = max(t_peak - temp_halfwidth, 50.0)
        temp2   = t_peak + temp_halfwidth
        temp_range = [temp1, temp2]
        logging.info(
            "Auto temp range: peak BT = %.1f K → [%.1f, %.1f] K",
            t_peak, temp1, temp2,
        )
    else:
        temp_range = sorted(temp_range)

    temp1, temp2 = temp_range

    # -------------------------------------------------------------------------
    # Stage 1 — NNLS on temperature grid
    # -------------------------------------------------------------------------
    temp_grid = np.arange(temp1, temp2 + temp_step, temp_step)
    dw_fit    = dw_rad[fit_mask]
    A = np.column_stack([utils.rad(wn_fit, T) - dw_fit for T in temp_grid])

    nnls_weights, _ = nnls(A, data_fit)
    model_nnls = A @ nnls_weights

    peak_idx, _ = find_peaks(
        nnls_weights,
        height=0.01 * nnls_weights.max() if nnls_weights.max() > 0 else 0.0,
        distance=3,
    )
    if len(peak_idx) < n_bb:
        peak_idx = np.argsort(nnls_weights)[-n_bb:][::-1]
    else:
        peak_idx = peak_idx[np.argsort(nnls_weights[peak_idx])[-n_bb:][::-1]]

    seed_temps = temp_grid[peak_idx]
    seed_fracs = nnls_weights[peak_idx] / nnls_weights[peak_idx].sum()

    logging.info(
        "NNLS stage — seed temperatures: %s K,  fractions: %s",
        [f"{t:.1f}" for t in seed_temps],
        [f"{f:.3f}" for f in seed_fracs],
    )

    # -------------------------------------------------------------------------
    # Stage 2 — Convex-hull seeding + strict upper-bound enforcement
    # -------------------------------------------------------------------------
    p0     = []
    bounds = [[], []]
    for i in range(n_bb):
        p0 += [float(seed_temps[i]), float(seed_fracs[i])]
        bounds[0] += [temp1, 0.0]
        bounds[1] += [temp2, 1.0]

    def fwd(wn_arr: np.ndarray, *p: float) -> np.ndarray:
        dw_arr = np.interp(wn_arr, wn0, dw_rad)
        bb_rad = np.zeros(len(wn_arr))
        for j in range(n_bb):
            bb_rad += p[2 * j + 1] * (utils.rad(wn_arr, p[2 * j]) - dw_arr)
        return bb_rad

    # Seed fit set from upper convex hull of the corrected data
    hull_mask = _upper_hull_mask(wn_fit, data_fit)
    logging.info(
        "Upper convex hull: %d / %d points selected as initial fit set",
        hull_mask.sum(), len(wn_fit),
    )

    popt = np.array(p0)
    pcov = np.full((len(p0), len(p0)), np.nan)
    perr = np.full(len(p0), np.nan)
    nerr = 0
    nfev = 0
    model_fit  = model_nnls.copy()   # used for violation check before first fit
    eff_weight = violation_weight    # escalates when hull stops growing
    n_escalations = 0

    for iteration in range(max_iter):
        wn_h    = wn_fit[hull_mask]
        data_h  = data_fit[hull_mask]

        if len(wn_h) < len(p0) + 1:
            logging.warning(
                "Hull2 fit: only %d points for %d parameters — stopping at iter %d",
                len(wn_h), len(p0), iteration,
            )
            break

        # Upweight hull points that the previous model already violated.
        model_h = fwd(wn_h, *popt)
        sigma_h = np.where(data_h > model_h * (1.0 + violation_tol),
                           1.0 / eff_weight, 1.0)

        try:
            popt, pcov, infodict, _, ier = curve_fit(
                fwd, wn_h, data_h,
                p0=popt, bounds=bounds,
                sigma=sigma_h, absolute_sigma=False,
                full_output=True, maxfev=5000,
            )
            perr  = np.sqrt(np.diag(pcov))
            nerr  = int(ier) if isinstance(ier, (int, np.integer)) else 1
            nfev += infodict["nfev"]
        except Exception as exc:
            logging.warning("Hull2 fit: curve_fit failed at iteration %d: %s", iteration, exc)
            break

        # Evaluate on the full fitting range and find violations.
        model_fit = fwd(wn_fit, *popt)
        viol_mask = data_fit > model_fit * (1.0 + violation_tol)
        n_viol    = int(viol_mask.sum())

        logging.info(
            "Hull2 iter %d — %d violations,  hull set: %d / %d channels,  eff_weight=%.1f",
            iteration + 1, n_viol, hull_mask.sum(), len(wn_fit), eff_weight,
        )

        if n_viol == 0:
            logging.info("Hull2 fit converged after %d iterations — no violations", iteration + 1)
            break

        new_hull_mask = hull_mask | viol_mask
        if np.array_equal(new_hull_mask, hull_mask):
            # Hull set is not growing; escalate weight and retry before giving up.
            if n_escalations < max_escalations:
                eff_weight  *= escalation_factor
                n_escalations += 1
                logging.info(
                    "Hull2: hull stalled with %d violations — escalating weight to %.1f (attempt %d/%d)",
                    n_viol, eff_weight, n_escalations, max_escalations,
                )
            else:
                logging.warning(
                    "Hull2: %d violations remain after %d escalations — stopping",
                    n_viol, n_escalations,
                )
                break
        else:
            # New violations found; reset escalation and extend the hull.
            eff_weight    = violation_weight
            n_escalations = 0
            hull_mask     = new_hull_mask

    # Final evaluation
    wn_hull    = wn_fit[hull_mask]
    data_hull  = data_fit[hull_mask]
    model_hull = fwd(wn_hull, *popt)
    model0     = fwd(wn0, *popt)

    # Emissivity — NEM-analogous formula: (data - dw) / model0
    # model0 is fit to (data-dw)/max_emiss, so (data-dw)/model0 ≈ max_emiss
    # at hull channels, matching NEM's max-emissivity convention.
    data_corr = data - dw_rad
    emiss = np.where(model0 > 0.0, data_corr / model0, 0.0)
    emiss[data_corr <= 0.0] = 0.0

    n_violations_final = int((data_fit > model_fit).sum())

    if not np.any(np.isnan(perr)):
        model0_up  = fwd(wn0, *(popt + perr))
        model0_dn  = fwd(wn0, *(popt - perr))
        model0_err = [model0_up - model0, model0 - model0_dn]
        with np.errstate(invalid='ignore', divide='ignore'):
            emiss_err = [
                np.where(model0 > 0.0, data_corr / (model0 + model0_err[0]) - emiss, 0.0),
                np.where(model0 > 0.0, emiss - data_corr / (model0 - model0_err[1]), 0.0),
            ]
    else:
        model0_err = [np.zeros_like(model0), np.zeros_like(model0)]
        emiss_err  = [np.zeros_like(emiss),  np.zeros_like(emiss)]

    bb_temps     = [popt[2 * j]     for j in range(n_bb)]
    bb_temps_err = [perr[2 * j]     for j in range(n_bb)]
    bb_fracs_raw = np.array([popt[2 * j + 1] for j in range(n_bb)])
    bb_fracs_err = [perr[2 * j + 1] for j in range(n_bb)]
    bb_fracs     = bb_fracs_raw / bb_fracs_raw.sum() if bb_fracs_raw.sum() > 0 else bb_fracs_raw
    mean_temp    = float(np.average(bb_temps, weights=bb_fracs))

    r2    = r2_score(data_hull, model_hull)                   if len(data_hull) > 1 else np.nan
    r20   = r2_score(data0[fit_mask], fwd(wn_fit, *popt))     if fit_mask.sum() > 1 else np.nan
    rmse  = root_mean_squared_error(data_hull, model_hull)
    rmse0 = root_mean_squared_error(data0, model0)

    stop = time.perf_counter()

    out = {
        "wn0":               wn0,
        "data0":             data0,
        "data0_eff":         data0_eff,
        "wn":                wn_hull,
        "data":              data_hull,
        "model":             model_hull,
        "model_err":         [fwd(wn_hull, *(popt + perr)) - model_hull,
                              model_hull - fwd(wn_hull, *(popt - perr))]
                             if not np.any(np.isnan(perr)) else
                             [np.zeros_like(model_hull), np.zeros_like(model_hull)],
        "res":               (model_hull - data_hull) / data_hull,
        "model0":            model0,
        "model0_err":        model0_err,
        "res0":              (model0 - data0_eff) / data0_eff,
        "emiss":             emiss,
        "emiss_err":         emiss_err,
        "fwd":               fwd,
        "n_bb":              n_bb,
        "p0":                p0,
        "bounds":            bounds,
        "temp_range":        temp_range,
        "temp_halfwidth":    temp_halfwidth,
        "wn_range":          wn_range,
        "co2_range":         co2_range,
        "temp_step":         temp_step,
        "violation_weight":  violation_weight,
        "violation_tol":     violation_tol,
        "escalation_factor": escalation_factor,
        "max_escalations":   max_escalations,
        "max_emiss":         max_emiss,
        "downwelling_t":     downwelling_t,
        "downwelling_e":     downwelling_e,
        "downwelling_rad":   downwelling_rad,
        "popt":              popt,
        "pcov":              pcov,
        "perr":              perr,
        "nerr":              nerr,
        "nfev":              nfev,
        "r2":                r2,
        "r20":               r20,
        "rmse":              rmse,
        "rmse0":             rmse0,
        "bb_temps":          bb_temps,
        "bb_temps_err":      bb_temps_err,
        "bb_fracs":          bb_fracs,
        "bb_fracs_err":      bb_fracs_err,
        "temp":              mean_temp,
        "rad_bb":            model0 * max_emiss + dw_rad,
        "temp_grid":         temp_grid,
        "nnls_weights":      nnls_weights,
        "n_violations_final": n_violations_final,
        "elapsed":           stop - start,
    }

    if plotout:
        fig, axes = plt.subplots(1, 2, figsize=[14, 5])
        if reverse:
            axes[0].invert_xaxis()
            axes[1].invert_xaxis()

        def fw(x):
            return 10000 / x

        axes[0].plot(wn0, data0_eff, c="k",        lw=1.5, label="Data (dw-corrected / max_emiss)")
        axes[0].plot(wn0, model0,    c="royalblue", lw=1.5,
                     label=f"Model ({n_bb} BBs)\nMean T = {mean_temp:.1f} K")
        axes[0].fill_between(
            wn0,
            model0 + 2 * model0_err[0],
            model0 - 2 * model0_err[1],
            color="royalblue", alpha=0.2, zorder=0, label=r"2$\sigma$",
        )
        axes[0].scatter(wn_hull, data_hull, c="tomato", s=4, zorder=5,
                        label=f"Hull+violations fit set ({len(wn_hull)} pts)")
        if n_violations_final > 0:
            viol_mask_plot = data_fit > model_fit
            axes[0].scatter(wn_fit[viol_mask_plot], data_fit[viol_mask_plot],
                            c="orange", s=8, zorder=6,
                            label=f"Residual violations ({n_violations_final})")
        axes[0].set(xlabel="Wavenumber [cm⁻¹]",
                    ylabel=fr"Spectral Radiance [$W / (m^2 \cdot sr \cdot cm^{{-1}})$]")
        axes[0].legend(title=f"max_emiss = {max_emiss:.2f}", fontsize=8)
        secax = axes[0].secondary_xaxis("top", functions=(fw, fw))
        secax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50])
        secax.set_xlabel(r"Wavelength [$\mu$m]")

        axes[1].plot(wn0, emiss, c="royalblue", lw=1.5, label="Emissivity")
        axes[1].fill_between(
            wn0,
            emiss - 2 * emiss_err[0],
            emiss + 2 * emiss_err[1],
            color="royalblue", alpha=0.2, zorder=0, label=r"2$\sigma$",
        )
        axes[1].axhline(1.0, c="k", ls="--", lw=0.8)
        axes[1].set(xlabel="Wavenumber [cm⁻¹]", ylabel="Emissivity")
        axes[1].legend(fontsize=8)
        secax = axes[1].secondary_xaxis("top", functions=(fw, fw))
        secax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50])
        secax.set_xlabel(r"Wavelength [$\mu$m]")

        fig.tight_layout()
        plt.show()

    return out


# =============================================================================
# ================================ hullfit2 ===================================
# =============================================================================

def emissivity_hullfit_linear(
    wn: np.ndarray,
    data: np.ndarray,
    temp_range: list[float] | None = None,
    temp_halfwidth: float = 50.0,
    temp_step: float = 5.0,
    wn_range: tuple[float, float] | None = (500.0, 1700.0),
    co2_range: tuple[float, float] | None = None,
    max_emiss: float = 1.0,
    downwelling_t: float = 0.0,
    downwelling_e: float = 1.0,
    downwelling_rad: np.ndarray | None = None,
    plotout: bool = False,
    reverse: bool = False,
) -> dict:
    """
    Fast two-temperature emissivity retrieval via NNLS seeding and a
    closed-form convex-mixture solve.  n_bb = 2 is fixed.

    Replaces the iterative curve_fit / violation-repair loop of
    emissivity_hullfit with a single closed-form LP solution: given T_cold and
    T_warm discovered by NNLS, the optimal cold-component fraction w1 that
    minimises total model radiance subject to model >= data_effective at every
    fitting channel is::

        w1 = clip( nanmin_j[ (P2_j - target_j) / (P2_j - P1_j) ],  0,  1 )

    where P1, P2 are the net (downwelling-corrected) Planck curves at T_cold
    and T_warm.  This is the exact LP solution for two components and requires
    no iteration, giving a guaranteed upper-bounding model at every channel.

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹).
    data : np.ndarray
        Calibrated sample radiance, same length as wn.
    temp_range : list[float] or None
        Explicit [T_min, T_max] bounds (K) for the NNLS grid.  When None,
        derived automatically as [T_nem - temp_halfwidth, T_nem + temp_halfwidth].
    temp_halfwidth : float
        Half-width (K) of the auto-derived temperature range.
    temp_step : float
        Temperature grid spacing (K) for the NNLS stage.
    wn_range : tuple[float, float] or None
        Fitting wavenumber window (wn_min, wn_max) cm⁻¹.  None uses full axis.
    co2_range : tuple[float, float] or None
        Wavenumber interval to exclude (CO₂ band), cm⁻¹.
    max_emiss : float
        Maximum emissivity normalisation applied before fitting.
    downwelling_t : float
        Downwelling air temperature (K).  0.0 disables correction.
    downwelling_e : float
        Downwelling emissivity fraction.
    downwelling_rad : np.ndarray or None
        Pre-computed downwelling radiance array.  Takes precedence over the
        scalar downwelling_t / downwelling_e pair.
    plotout : bool
        If True, display a diagnostic figure.
    reverse : bool
        If True, reverse the wavenumber axis in diagnostic plots.

    Returns
    -------
    dict
        Keys: ``wn0``, ``data0``, ``data0_eff``, ``wn_fit``, ``model0``,
        ``emiss``, ``temp``, ``rad_bb``, ``bb_temps``, ``bb_fracs``,
        ``wn_range``, ``co2_range``, ``temp_range``, ``temp_step``,
        ``max_emiss``, ``temp_grid``, ``nnls_weights``, ``elapsed``.
    """
    start = time.perf_counter()

    wn0   = wn.copy()
    data0 = data.copy() / max_emiss

    if downwelling_t == 0.0 and downwelling_rad is None:
        dw_rad = np.zeros(len(wn0))
    else:
        if downwelling_rad is None:
            downwelling_rad = downwelling_e * utils.rad(wn0, downwelling_t)
        dw_rad = downwelling_rad

    data0_eff = (data - dw_rad) / max_emiss

    if wn_range is not None:
        wn1, wn2 = wn_range
        fit_mask = (wn0 >= wn1) & (wn0 <= wn2)
    else:
        fit_mask = np.ones(len(wn0), dtype=bool)
    if co2_range is not None:
        fit_mask &= (wn0 < co2_range[0]) | (wn0 > co2_range[1])
    wn_fit   = wn0[fit_mask]
    data_fit = data0_eff[fit_mask]
    dw_fit   = dw_rad[fit_mask]

    # Peak brightness temperature — used for auto temp range and feasibility clamp.
    # The LP constraint is P2 = rad(T_warm) - dw >= data_fit, so rad(T_warm) >= data_fit + dw.
    # Use BBT of (data_fit + dw_fit) to set the correct temperature floor.
    feasibility_data = data_fit + dw_fit   # = data[fit_mask] / max_emiss when max_emiss=1
    bt_fit = utils.bbt(wn_fit, np.where(feasibility_data > 0, feasibility_data, np.nan))
    t_nem  = float(bt_fit[np.isfinite(bt_fit)].max())

    if temp_range is None:
        temp1 = max(t_nem - temp_halfwidth, 50.0)
        temp2 = t_nem + temp_halfwidth
        temp_range = [temp1, temp2]
        logging.info(
            "hullfit2: auto temp range: T_nem = %.1f K → [%.1f, %.1f] K",
            t_nem, temp1, temp2,
        )
    else:
        temp_range = sorted(temp_range)

    temp1, temp2 = temp_range

    # NNLS on temperature grid → 2 dominant peaks.
    temp_grid    = np.arange(temp1, temp2 + temp_step, temp_step)
    A            = np.column_stack([utils.rad(wn_fit, T) - dw_fit for T in temp_grid])
    nnls_weights, _ = nnls(A, data_fit)

    peak_idx, _ = find_peaks(
        nnls_weights,
        height=0.01 * nnls_weights.max() if nnls_weights.max() > 0 else 0.0,
        distance=3,
    )
    if len(peak_idx) < 2:
        peak_idx = np.argsort(nnls_weights)[-2:][::-1]
    else:
        peak_idx = peak_idx[np.argsort(nnls_weights[peak_idx])[-2:][::-1]]

    T_cold = float(np.sort(temp_grid[peak_idx])[0])

    # Optimise T_warm over [t_nem, temp2] by minimising total model radiance
    # subject to the closed-form LP constraint at each candidate temperature.
    # The NNLS warm peak overshoots t_nem because the grid extends above it;
    # curve_fit would pull it back to the optimal value, which this 1-D search
    # recovers without the full nonlinear fit.
    P1     = utils.rad(wn_fit, T_cold) - dw_fit
    target = data_fit

    def _total_radiance(T_w: float) -> float:
        if T_w <= T_cold:
            return np.inf
        P2_c  = utils.rad(wn_fit, T_w) - dw_fit
        denom = np.where(np.abs(P2_c - P1) > 0.0, P2_c - P1, np.nan)
        w1_b  = (P2_c - target) / denom
        if np.nanmin(w1_b) < 0.0:
            return np.inf           # infeasible — T_w too low
        w1_c  = float(np.clip(np.nanmin(w1_b), 0.0, 1.0))
        return float(np.sum(w1_c * P1 + (1.0 - w1_c) * P2_c))

    opt    = minimize_scalar(_total_radiance, bounds=(t_nem, temp2), method='bounded')
    T_warm = float(opt.x)

    logging.info(
        "hullfit2: T_cold = %.1f K (NNLS),  T_warm = %.1f K (1-D opt,  %d evals)",
        T_cold, T_warm, opt.nfev,
    )

    # Closed-form LP for w1 (cold-component fraction).
    # P1 and target already computed above; only P2 needs the final T_warm.
    P2 = utils.rad(wn_fit, T_warm) - dw_fit

    denom       = np.where(np.abs(P2 - P1) > 0.0, P2 - P1, np.nan)
    w1_per_band = (P2 - target) / denom
    w1 = float(np.clip(np.nanmin(w1_per_band), 0.0, 1.0))
    w2 = 1.0 - w1

    logging.info("hullfit2: w_cold = %.4f,  w_warm = %.4f", w1, w2)

    # Model over full axis.
    P1_full = utils.rad(wn0, T_cold) - dw_rad
    P2_full = utils.rad(wn0, T_warm) - dw_rad
    model0  = w1 * P1_full + w2 * P2_full

    # Emissivity.
    data_corr = data - dw_rad
    emiss = np.where(model0 > 0.0, data_corr / model0, 0.0)
    emiss[data_corr <= 0.0] = 0.0

    mean_temp = w1 * T_cold + w2 * T_warm

    stop = time.perf_counter()

    out = {
        'wn0':          wn0,
        'data0':        data0,
        'data0_eff':    data0_eff,
        'wn_fit':       wn_fit,
        'model0':       model0,
        'emiss':        emiss,
        'temp':         mean_temp,
        'rad_bb':       model0 * max_emiss + dw_rad,
        'bb_temps':     [T_cold, T_warm],
        'bb_fracs':     [w1, w2],
        'wn_range':     wn_range,
        'co2_range':    co2_range,
        'temp_range':   temp_range,
        'temp_step':    temp_step,
        'max_emiss':    max_emiss,
        'temp_grid':    temp_grid,
        'nnls_weights': nnls_weights,
        'elapsed':      stop - start,
    }

    if plotout:
        fig, axes = plt.subplots(1, 2, figsize=[14, 5])
        if reverse:
            axes[0].invert_xaxis()
            axes[1].invert_xaxis()

        def fw(x):
            return 10000 / x

        axes[0].plot(wn0, data0_eff, c='k',          lw=1.5, label='Data (dw-corrected / max_emiss)')
        axes[0].plot(wn0, model0,    c='darkorange',  lw=1.5,
                     label=f'hullfit2  T_cold={T_cold:.1f} K  T_warm={T_warm:.1f} K\n'
                           f'w_cold={w1:.3f}  w_warm={w2:.3f}  mean T={mean_temp:.1f} K')
        axes[0].set(xlabel='Wavenumber [cm⁻¹]',
                    ylabel=r'Spectral Radiance [$W / (m^2 \cdot sr \cdot cm^{-1})$]')
        axes[0].legend(title=f'max_emiss = {max_emiss:.2f}', fontsize=8)
        secax = axes[0].secondary_xaxis('top', functions=(fw, fw))
        secax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50])
        secax.set_xlabel(r'Wavelength [$\mu$m]')

        axes[1].plot(wn0, emiss, c='darkorange', lw=1.5, label='Emissivity')
        axes[1].axhline(1.0, c='k', ls='--', lw=0.8)
        axes[1].set(xlabel='Wavenumber [cm⁻¹]', ylabel='Emissivity')
        axes[1].legend(fontsize=8)
        secax = axes[1].secondary_xaxis('top', functions=(fw, fw))
        secax.xaxis.set_ticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50])
        secax.set_xlabel(r'Wavelength [$\mu$m]')

        fig.tight_layout()
        plt.show()

    return out


# =============================================================================
# ================================ dehyd ======================================
# =============================================================================

# Path to the shipped reference water emissivity spectrum.
# Ref: spectral_tools.dvrc::dehyd2, $DV_SCRIPT_FILES/dehyd_water.txt
_DEHYD_WATER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_data', 'dehyd_water.txt')

# Wavenumber pairs derived from DaVinci channel indices (1-indexed) on the
# dehyd_water.txt grid (198.63–3999.71 cm⁻¹, step ≈ 1.929 cm⁻¹).
# Low group:  H₂O pure-rotational band       (~220–351 cm⁻¹)
# High group: H₂O ν₂ bending mode            (~1413–1734 cm⁻¹)
# Ref: spectral_tools.dvrc::dehyd2, ~lines 4351–4362
_DEHYD_LOW_ABSORB  = np.array([227.6, 246.8, 254.6, 277.7, 279.6, 302.8, 324.0, 327.8, 351.0])
_DEHYD_LOW_CONT    = np.array([219.8, 239.1, 258.4, 271.9, 285.4, 306.6, 320.1, 331.7, 347.1])
_DEHYD_HIGH_ABSORB = np.array([1419.4, 1436.7, 1456.0, 1506.2, 1521.6, 1540.9, 1558.2,
                                1652.7, 1683.6, 1700.9, 1716.4, 1733.7])
_DEHYD_HIGH_CONT   = np.array([1413.6, 1440.6, 1452.2, 1502.3, 1513.9, 1546.7, 1564.0,
                                1658.5, 1679.7, 1691.3, 1710.6, 1726.0])


def dehyd(
    wn: np.ndarray,
    emiss: np.ndarray,
    water_ref_path: str | None = None,
) -> dict:
    """
    Remove residual water vapour features from a lab emissivity spectrum.

    Direct port of DaVinci's ``dehyd2()``
    (spectral_tools.dvrc::dehyd2, lines 4338–4385), generalised to work in
    wavenumber space rather than fixed channel indices.

    The algorithm scales a reference pure-water emissivity spectrum by the
    relative water-vapour content of the sample, measured via
    absorption-band / continuum ratios at known water-sensitive wavenumbers,
    then divides the sample emissivity by the resulting correction spectrum.

    Two spectral groups are used when coverage permits:

    * **Low group** (~220–351 cm⁻¹): H₂O pure-rotational band.
      Applied only when ``wn`` covers this far-IR region.
    * **High group** (~1413–1734 cm⁻¹): H₂O ν₂ bending mode.
      Applied for all standard MIR instruments; a warning is raised if absent.

    Parameters
    ----------
    wn : np.ndarray
        Wavenumber axis (cm⁻¹), shape (n_bands,).
    emiss : np.ndarray
        Emissivity spectrum to correct, shape (n_bands,).
    water_ref_path : str or None
        Path to the two-column (wavenumber, emissivity) reference water
        spectrum in DaVinci VM text format (3-line header).
        Defaults to the shipped ``dehyd_water.txt``.

    Returns
    -------
    dict
        Keys:

        ``emiss``
            Corrected emissivity, shape (n_bands,).
        ``emiss_orig``
            Input emissivity before correction (copy), shape (n_bands,).
        ``water_ref``
            Reference water emissivity resampled to ``wn``, shape (n_bands,).
        ``water_index_ref``
            Water absorption index of the reference spectrum.
        ``water_index``
            Water absorption index of the sample.
        ``water_ratio``
            ``water_index / water_index_ref``; the scaling factor applied.
        ``fix``
            Multiplicative correction spectrum (``emiss / fix = corrected``).
        ``low_available``
            Whether the low-wavenumber group was used.
        ``high_available``
            Whether the high-wavenumber group was used.
    """
    # Ref: spectral_tools.dvrc::dehyd2, lines 4338–4385
    path = water_ref_path if water_ref_path is not None else _DEHYD_WATER_PATH
    ref_data  = np.loadtxt(path, skiprows=3)
    ref_wn    = ref_data[:, 0]   # increasing, 198–3999 cm⁻¹
    ref_emiss = ref_data[:, 1]

    # Resample reference onto sample wn axis (np.interp requires xp increasing)
    water_ref = np.interp(wn, ref_wn, ref_emiss)

    # Determine group coverage
    low_available  = wn.min() <= _DEHYD_LOW_ABSORB.max()
    high_available = wn.max() >= _DEHYD_HIGH_ABSORB.min()

    if not low_available and not high_available:
        raise ValueError(
            f"dehyd: no water-band pairs overlap the measured range "
            f"({wn.min():.0f}–{wn.max():.0f} cm⁻¹)"
        )
    if not low_available:
        logging.info(
            "dehyd: low group (220–351 cm⁻¹) not covered — using high group only "
            "(normal for standard MIR instruments)"
        )
    if not high_available:
        logging.warning("dehyd: high group (1413–1734 cm⁻¹) not covered — using low group only")

    def _group_mean_ratio(
        spectrum: np.ndarray,
        absorb_wns: np.ndarray,
        cont_wns: np.ndarray,
    ) -> float:
        # Nearest-neighbour lookup in wavenumber space
        ratios = [
            spectrum[np.argmin(np.abs(wn - a))] / spectrum[np.argmin(np.abs(wn - c))]
            for a, c in zip(absorb_wns, cont_wns)
        ]
        return float(np.mean(ratios))

    # Water index = departure of absorbing channels from local continuum.
    # Each available group contributes one mean ratio; groups are averaged
    # with equal weight, matching DaVinci's *0.5 for the two-group case.
    # Ref: spectral_tools.dvrc::dehyd2, ~lines 4351–4362
    ref_group_means  = []
    samp_group_means = []
    if low_available:
        ref_group_means.append(_group_mean_ratio(water_ref, _DEHYD_LOW_ABSORB,  _DEHYD_LOW_CONT))
        samp_group_means.append(_group_mean_ratio(emiss,    _DEHYD_LOW_ABSORB,  _DEHYD_LOW_CONT))
    if high_available:
        ref_group_means.append(_group_mean_ratio(water_ref, _DEHYD_HIGH_ABSORB, _DEHYD_HIGH_CONT))
        samp_group_means.append(_group_mean_ratio(emiss,    _DEHYD_HIGH_ABSORB, _DEHYD_HIGH_CONT))

    wat_index    = 1.0 - float(np.mean(ref_group_means))
    sample_index = 1.0 - float(np.mean(samp_group_means))

    if abs(wat_index) < 1e-6:
        logging.warning("dehyd: reference water index near zero — skipping correction")
        ratio = 0.0
    else:
        ratio = sample_index / wat_index

    if ratio < 0.0:
        logging.warning("dehyd: negative water ratio (%.3f) — clamping to 0", ratio)
        ratio = 0.0
    elif ratio > 5.0:
        logging.warning("dehyd: large water ratio (%.3f) — clamping to 5.0", ratio)
        ratio = 5.0

    logging.info(
        "dehyd: water_index=%.4f  ref_index=%.4f  ratio=%.3f",
        sample_index, wat_index, ratio,
    )

    # Correction spectrum; clamp to avoid division by zero or sign flip
    # in noisy far-IR channels.
    # Ref: spectral_tools.dvrc::dehyd2, ~lines 4372–4374
    fix = 1.0 - (1.0 - water_ref) * ratio
    fix = np.maximum(fix, 1e-6)

    return {
        'emiss':           emiss / fix,
        'emiss_orig':      emiss.copy(),
        'water_ref':       water_ref,
        'water_index_ref': wat_index,
        'water_index':     sample_index,
        'water_ratio':     ratio,
        'fix':             fix,
        'low_available':   low_available,
        'high_available':  high_available,
    }



def insert_plot_gaps(
    x: np.ndarray,
    y: np.ndarray,
    threshold_factor: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert NaN pairs into (x, y) at large gaps in x so that matplotlib
    renders them as breaks in the line rather than spanning the gap.

    A gap is any step between consecutive x values that exceeds
    threshold_factor × median_step.  Suitable for display only — the
    returned arrays must not be used for computation.

    Parameters
    ----------
    x : np.ndarray
        Monotonic xaxis values (cm⁻¹ or µm).
    y : np.ndarray
        Spectral data, same length as x.
    threshold_factor : float
        Multiplier on the median step above which a gap is flagged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (x_plot, y_plot) with NaN pairs inserted at each detected gap.
    """
    if len(x) < 2:
        return x, y

    diffs      = np.abs(np.diff(x))
    median_step = np.median(diffs)
    gap_idxs   = np.where(diffs > threshold_factor * median_step)[0]

    if len(gap_idxs) == 0:
        return x, y

    # Build output by inserting a NaN pair after each gap index
    nan = np.array([np.nan])
    x_parts, y_parts = [], []
    prev = 0
    for idx in gap_idxs:
        x_parts.extend([x[prev:idx + 1], nan])
        y_parts.extend([y[prev:idx + 1], nan])
        prev = idx + 1
    x_parts.append(x[prev:])
    y_parts.append(y[prev:])

    return np.concatenate(x_parts), np.concatenate(y_parts)


def resample_spectrum(
    src_wn: np.ndarray,
    src_data: np.ndarray,
    target_xaxis: np.ndarray,
    target_is_wl: bool = False,
) -> np.ndarray:
    """
    Interpolate a single spectrum onto a target xaxis grid.

    Parameters
    ----------
    src_wn : np.ndarray
        Source wavenumber axis (cm⁻¹). May be ascending or descending.
    src_data : np.ndarray
        Source spectral data (emissivity), same length as src_wn.
    target_xaxis : np.ndarray
        Target grid. If target_is_wl=True, values are in µm (ascending).
        Otherwise values are in cm⁻¹ (any monotonic order).
    target_is_wl : bool
        If True, convert the source axis from cm⁻¹ to µm before
        interpolating. Output corresponds to target_xaxis in µm.

    Returns
    -------
    np.ndarray
        Resampled spectral data, same shape as target_xaxis.
        NaN outside the source range.
    """
    if target_is_wl:
        # wn → µm reverses direction; flip to get ascending wavelength axis
        src_x = np.flip(1e4 / src_wn)
        src_d = np.flip(src_data)
    else:
        src_x = src_wn.copy()
        src_d = src_data

    # np.interp requires ascending src_x
    if src_x[-1] < src_x[0]:
        src_x = src_x[::-1]
        src_d = src_d[::-1]

    return np.interp(target_xaxis, src_x, src_d, left=np.nan, right=np.nan)


_WL_INSTRUMENTS = {'themis', 'aster', 'master', 'tims'}

# Canonical instrument key → source DV filename
_INSTRUMENT_FILES = {
    'tessingle': 'ASU_speclib_TESsingle_DV.hdf',
    'tesdouble': 'ASU_speclib_TESdouble_DV.hdf',
    'tes73':     'ASU_speclib_TES73_DV.hdf',
    'minites':   'ASU_speclib_miniTES_DV.hdf',
    'microlab':  'ASU_speclib_microlab_DV.hdf',
    'themis':    'ASU_speclib_THEMIS_DV.hdf',
    'aster':     'ASU_speclib_ASTER_DV.hdf',
    'master':    'ASU_speclib_MASTER_DV.hdf',
    'tims':      'ASU_speclib_TIMS_DV.hdf',
}


_REFERENCE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reference_data')


def save_instrument_grids(
    dst_path: str | None = None,
    speclib_dir: str = './ASU_speclib',
) -> None:
    """
    Extract instrument xaxis grids from DV HDF files and save to a .npz file.

    Run once to generate the compact grid file.  After that,
    load_instrument_grids() reads from the .npz rather than the full HDF files.

    Parameters
    ----------
    dst_path : str or None
        Destination path for the .npz file.  Defaults to
        ``reference_data/instrument_grids.npz`` inside the package directory.
    speclib_dir : str
        Directory containing the DV HDF source files.
    """
    if dst_path is None:
        dst_path = os.path.join(_REFERENCE_DATA_DIR, 'instrument_grids.npz')
    arrays = {}
    for name, fname in _INSTRUMENT_FILES.items():
        path = f'{speclib_dir}/{fname}'
        try:
            arrays[name] = utils.readHDF(path)['xaxis']
        except Exception as exc:
            logging.warning(f'load_instrument_grids: skipping {name} ({exc})')

    np.savez(dst_path, **arrays)
    logging.info(f'Instrument grids saved to {dst_path} '
                 f'({list(arrays.keys())})')


def load_instrument_grids(
    grids_path: str | None = None,
) -> dict:
    """
    Load instrument xaxis grids from the compact .npz file produced by
    save_instrument_grids().

    Parameters
    ----------
    grids_path : str or None
        Path to the .npz file.  Defaults to
        ``reference_data/instrument_grids.npz`` inside the package directory.

    Returns
    -------
    dict
        {name: {'xaxis': np.ndarray, 'is_wl': bool, 'xlabel': str}}
        name is the lowercase instrument key (e.g. 'tessingle').
        is_wl=True means the xaxis is in µm (wavelength).
    """
    if grids_path is None:
        grids_path = os.path.join(_REFERENCE_DATA_DIR, 'instrument_grids.npz')
    data   = np.load(grids_path)
    grids  = {}
    for name in data.files:
        is_wl = name in _WL_INSTRUMENTS
        grids[name] = {
            'xaxis':  data[name],
            'is_wl':  is_wl,
            'xlabel': 'Wavelength (\u00b5m)' if is_wl
                      else 'Wavenumber (cm\u207b\xb9)',
        }
    return grids


# =============================================================================
# ================================= merge =====================================
# =============================================================================

def _detect_merge_type(d: dict) -> str:
    """
    Identify the structural type of an output dict for merging.

    Returns
    -------
    'per_entry'
        All top-level values are dicts (spectral library / album format).
    'measurement'
        Top-level ``xaxis`` key present (emcal3 / measurement format).

    Raises
    ------
    ValueError
        If neither condition is met.
    """
    if 'xaxis' in d:
        return 'measurement'
    if d and all(isinstance(v, dict) for v in d.values()):
        return 'per_entry'
    raise ValueError(
        "Cannot determine dict type for merge: no top-level 'xaxis' key "
        "and not all top-level values are dicts."
    )


def _check_xaxis_compat(xaxes: list, label: str = '') -> np.ndarray:
    """
    Verify that all arrays in *xaxes* are shape- and value-identical.

    Parameters
    ----------
    xaxes : list of np.ndarray
        Spectral axis arrays to compare.
    label : str
        Context string for error messages.

    Returns
    -------
    np.ndarray
        The reference xaxis (first element).

    Raises
    ------
    ValueError
        If any xaxis does not match the first.
    """
    ref = np.asarray(xaxes[0])
    ctx = f" ({label})" if label else ''
    for i, ax in enumerate(xaxes[1:], start=2):
        ax = np.asarray(ax)
        if ax.shape != ref.shape or not np.allclose(ax, ref):
            raise ValueError(
                f"merge: xaxis mismatch{ctx} between input 1 and input {i} — "
                f"shapes {ref.shape} vs {ax.shape}."
            )
    return ref


def _build_common_xaxis(xaxes: list) -> np.ndarray:
    """
    Construct a common wavenumber grid from a list of xaxis arrays.

    The grid spans the intersection of all input ranges, sampled at the mean
    of all input spacings.  Direction (ascending / descending) is inherited
    from the first input.

    Parameters
    ----------
    xaxes : list of np.ndarray
        Wavenumber axes to reconcile.

    Returns
    -------
    np.ndarray
        New uniform xaxis covering the common range.

    Raises
    ------
    ValueError
        If the input ranges do not overlap.
    """
    arrays = [np.asarray(ax) for ax in xaxes]
    descending = arrays[0][0] > arrays[0][-1]

    wn_lo = max(ax.min() for ax in arrays)
    wn_hi = min(ax.max() for ax in arrays)
    if wn_lo >= wn_hi:
        raise ValueError(
            f"merge: xaxis ranges do not overlap "
            f"(intersection is [{wn_lo:.4f}, {wn_hi:.4f}] cm⁻¹)."
        )

    mean_spacing = np.mean([
        abs(float(ax[-1]) - float(ax[0])) / (len(ax) - 1)
        for ax in arrays
    ])
    n_pts = int(round((wn_hi - wn_lo) / mean_spacing)) + 1
    common = np.linspace(wn_lo, wn_hi, n_pts)

    logging.info(
        "merge: common xaxis [%.2f, %.2f] cm⁻¹, %d pts, "
        "mean spacing %.4f cm⁻¹.",
        wn_lo, wn_hi, n_pts, mean_spacing,
    )

    return common[::-1] if descending else common


def _resample_measurement_dict(d: dict, ref_xaxis: np.ndarray) -> dict:
    """
    Resample all spectral arrays in a measurement dict onto *ref_xaxis*.

    Arrays resampled per-spectrum using :func:`resample_spectrum`:

    * ``xaxis`` — replaced by *ref_xaxis*.
    * ``wl``    — recomputed as ``1e4 / ref_xaxis``.
    * ``data``  — (n_samples, n_bands) matrix resampled row-by-row.
    * Named sub-dicts (``emiss``, ``rad``, ``sbm``, ``rad0``) — each
      ``{label: (n_bands,)}`` value resampled independently.
    * All other keys (scalars, ``calib``, …) passed through unchanged.

    Parameters
    ----------
    d : dict
        Measurement-format dict to resample.
    ref_xaxis : np.ndarray
        Target wavenumber axis (cm⁻¹).

    Returns
    -------
    dict
        Copy of *d* with spectral arrays on *ref_xaxis*.
    """
    src_xaxis = np.asarray(d['xaxis'])
    n_src = len(src_xaxis)
    _SPECTRAL_SUB_DICTS = {'emiss', 'rad', 'sbm', 'rad0'}
    out = {}
    for key, val in d.items():
        if key == 'xaxis':
            out['xaxis'] = ref_xaxis
        elif key == 'wl':
            out['wl'] = 1e4 / ref_xaxis
        elif key == 'data' and isinstance(val, np.ndarray) and val.ndim == 2:
            out['data'] = np.vstack([
                resample_spectrum(src_xaxis, val[i], ref_xaxis)
                for i in range(val.shape[0])
            ])
        elif key in _SPECTRAL_SUB_DICTS and isinstance(val, dict):
            out[key] = {
                label: resample_spectrum(src_xaxis, np.asarray(arr), ref_xaxis)
                for label, arr in val.items()
            }
        elif key == 'emiss_full' and isinstance(val, dict):
            # Nested {label: {field: array}} — resample any spectral arrays
            resampled_full = {}
            for label, sub in val.items():
                if not isinstance(sub, dict):
                    resampled_full[label] = sub
                    continue
                resampled_sub = {}
                for field, arr in sub.items():
                    if (isinstance(arr, np.ndarray)
                            and arr.ndim == 1
                            and len(arr) == n_src):
                        resampled_sub[field] = resample_spectrum(
                            src_xaxis, arr, ref_xaxis)
                    else:
                        resampled_sub[field] = arr
                resampled_full[label] = resampled_sub
            out[key] = resampled_full
        else:
            out[key] = val
    return out


def _merge_sub_dicts(sub_dicts: list, field: str) -> dict:
    """
    Merge a list of ``{label: value}`` dicts, warning on duplicate labels.

    Duplicate labels that carry identical array data are silently deduplicated.
    Duplicates with differing data are stored under a suffixed key
    (``label_2``, ``label_3``, …).

    Parameters
    ----------
    sub_dicts : list of dict
        Dicts to merge; all have the same role (e.g. all are ``emiss`` dicts).
    field : str
        Field name used in warning messages.

    Returns
    -------
    dict
        Merged dict.
    """
    merged = {}
    for i, sd in enumerate(sub_dicts, start=1):
        for label, val in sd.items():
            if label not in merged:
                merged[label] = val
            else:
                existing = merged[label]
                # Silent dedup: identical arrays or scalars
                try:
                    if (isinstance(val, np.ndarray) and isinstance(existing, np.ndarray)
                            and np.array_equal(val, existing)):
                        continue
                    if val == existing:
                        continue
                except (TypeError, ValueError):
                    pass

                suffix_n = 2
                new_label = f"{label}_{suffix_n}"
                while new_label in merged:
                    suffix_n += 1
                    new_label = f"{label}_{suffix_n}"
                logging.warning(
                    "merge: duplicate label '%s' in '%s' from input %d "
                    "(differing data) — stored as '%s'.",
                    label, field, i, new_label,
                )
                merged[new_label] = val
    return merged


def _merge_per_entry(dicts: list) -> dict:
    """
    Merge per-entry (spectral library / album) dicts.

    Each top-level key is a spectrum identifier (string or int).  Duplicate
    identifiers with identical spectral data are silently skipped; those with
    differing data are kept under a suffixed key.  The ``xaxis`` arrays inside
    all entries must be mutually consistent.

    Parameters
    ----------
    dicts : list of dict
        Per-entry dicts to merge.

    Returns
    -------
    dict
        Merged per-entry dict containing all unique spectra.
    """
    # Validate xaxis consistency across all entries in all dicts
    all_xaxes = [
        np.asarray(v['xaxis'])
        for d in dicts
        for v in d.values()
        if isinstance(v, dict) and 'xaxis' in v
    ]
    if all_xaxes:
        _check_xaxis_compat(all_xaxes, label='per-entry xaxis')

    merged = {}
    for i, d in enumerate(dicts, start=1):
        for key, val in d.items():
            if key not in merged:
                merged[key] = val
                continue

            existing = merged[key]
            if not (isinstance(val, dict) and isinstance(existing, dict)):
                logging.warning(
                    "merge: duplicate key '%s' (non-dict) from input %d — skipped.", key, i
                )
                continue

            # Compare spectral data to distinguish true duplicates from conflicts
            existing_data = existing.get('data')
            new_data      = val.get('data')
            if (existing_data is not None and new_data is not None
                    and np.array_equal(existing_data, new_data)):
                logging.debug(
                    "merge: skipping duplicate entry '%s' from input %d (identical data).", key, i
                )
                continue

            suffix_n  = 2
            new_key   = f"{key}_{suffix_n}"
            while new_key in merged:
                suffix_n += 1
                new_key = f"{key}_{suffix_n}"
            logging.warning(
                "merge: entry '%s' from input %d has different data from an "
                "earlier input — stored as '%s'.", key, i, new_key
            )
            merged[new_key] = val

    return merged


def _sync_label_with_spectra(merged: dict) -> None:
    """
    Rebuild the ``label`` key in-place to match the merged spectral sub-dicts.

    Merging combines the label-keyed sub-dicts (``emiss``, ``rad``, …) by
    suffixing name collisions (``_2``, ``_3``, …), so the surviving set of
    sample labels is the *keys* of those sub-dicts — not the original ``label``
    vector, which may have been deduplicated (when inputs share identical
    label arrays) or carried over from a single input.  The GUI and downstream
    consumers enumerate samples via ``label`` and look each one up in ``emiss``,
    so the two must stay consistent.  The first present spectral sub-dict, in
    priority order, defines the canonical sample list and ordering.

    Parameters
    ----------
    merged : dict
        Merged measurement dict, modified in place.  No-op if it carries no
        recognised spectral sub-dict.
    """
    for key in ('emiss', 'rad', 'sbm', 'rad0'):
        sub = merged.get(key)
        if isinstance(sub, dict) and sub:
            # Keep 'label' a plain list to match emcal/load_sbm output; GUI and
            # other consumers index, slice, and truth-test it (an ndarray would
            # raise "truth value of an array is ambiguous" on `if labels:`).
            merged['label'] = [str(k) for k in sub.keys()]
            return


def _merge_measurement(dicts: list, resample: bool = False) -> dict:
    """
    Merge measurement-format (emcal3-style) dicts.

    All inputs must share an identical top-level ``xaxis`` unless *resample*
    is ``True``, in which case a common xaxis is constructed from the
    intersection of all input ranges at their mean sampling interval, and all
    inputs are resampled onto it.  Per-key strategy:

    * **Shared axis arrays** (``xaxis``, ``wl``, …): verified identical across
      all inputs, kept once.
    * **Stackable sample arrays** (``data``, …): concatenated along axis 0.
    * **Named-entry sub-dicts** (``emiss``, ``rad``, ``sbm``, ``rad0``,
      ``sample_temps``, …): merged label-by-label; duplicates warned + suffixed.
    * **Calibration sub-dict** (``calib``): kept from the first input; a
      warning is raised if subsequent inputs carry a different ``calib``.
    * **Scalar metadata**: kept from the first input; a warning is raised on
      disagreement.

    Parameters
    ----------
    dicts : list of dict
        Measurement-format dicts to merge.
    resample : bool, optional
        If ``True``, build a common xaxis (intersection of all input ranges,
        mean sampling interval) and resample all inputs onto it via linear
        interpolation.  Default ``False``.

    Returns
    -------
    dict
        Merged measurement dict.
    """
    if resample:
        common_xaxis = _build_common_xaxis([d['xaxis'] for d in dicts])
        dicts = [_resample_measurement_dict(d, common_xaxis) for d in dicts]

    xaxes     = [d['xaxis'] for d in dicts]
    ref_xaxis = _check_xaxis_compat(xaxes, label='top-level xaxis')

    all_keys = set()
    for d in dicts:
        all_keys.update(d.keys())

    merged = {'xaxis': ref_xaxis}

    for key in sorted(all_keys - {'xaxis'}):
        values = [d[key] for d in dicts if key in d]
        if len(values) < len(dicts):
            logging.warning(
                "merge: key '%s' absent from %d of %d inputs — merging available values.",
                key, len(dicts) - len(values), len(dicts),
            )
        first = values[0]

        if isinstance(first, dict):
            if key == 'calib':
                merged[key] = first
                for i, v in enumerate(values[1:], start=2):
                    if set(v.keys()) != set(first.keys()):
                        logging.warning(
                            "merge: 'calib' from input %d differs from input 1 — "
                            "keeping input 1's calib.", i
                        )
            elif key == 'notes':
                # notes is a column-keyed dict of parallel sequences — concatenate
                # each column across inputs.  Columns may be Python lists or NumPy
                # arrays, so coerce to list before concatenating (summing raw arrays
                # would trigger element-wise broadcasting instead of concatenation).
                cols: set[str] = set()
                for v in values:
                    cols.update(v.keys())
                merged[key] = {
                    col: [item for v in values for item in list(v.get(col, []))]
                    for col in sorted(cols)
                }
            else:
                merged[key] = _merge_sub_dicts(values, key)

        elif isinstance(first, np.ndarray):
            # Identical across all inputs → shared axis; otherwise stack.
            # Numeric arrays are compared with a tolerance; non-numeric arrays
            # (e.g. string labels) fall back to exact element-wise equality
            # since np.allclose cannot promote strings to a float tolerance.
            def _arrays_match(v: np.ndarray, ref: np.ndarray) -> bool:
                if v.shape != ref.shape:
                    return False
                if v.dtype.kind in 'fc' and ref.dtype.kind in 'fc':
                    return bool(np.allclose(v, ref, equal_nan=True))
                return bool(np.array_equal(v, ref))

            all_equal = all(_arrays_match(v, first) for v in values[1:])
            if all_equal:
                merged[key] = first
            elif first.ndim == 1:
                # Per-sample 1-D vectors (e.g. 'label') — concatenate end-to-end
                # so differing lengths across inputs are joined, not row-stacked.
                merged[key] = np.concatenate(values, axis=0)
            else:
                # 2-D+ sample matrices (n_spectra × n_channels) — stack rows.
                try:
                    merged[key] = np.concatenate(
                        [np.atleast_2d(v) for v in values], axis=0
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"merge: key '{key}' arrays differ and cannot be stacked "
                        f"(shapes: {[v.shape for v in values]}): {exc}"
                    ) from exc

        elif isinstance(first, list):
            # Concatenate lists (e.g. the 'label' key from emcal output)
            merged[key] = sum(values, [])

        elif isinstance(first, (int, float, str, bytes, np.generic)):
            if not all(v == first for v in values[1:]):
                logging.warning(
                    "merge: scalar key '%s' differs across inputs — keeping first value (%r).",
                    key, first,
                )
            merged[key] = first

        else:
            logging.warning(
                "merge: key '%s' has unsupported type %s — keeping first.", key, type(first)
            )
            merged[key] = first

    _sync_label_with_spectra(merged)
    return merged


def _detect_merge_direction(dicts: list, how: str) -> str:
    """
    Determine whether a measurement merge is vertical (same xaxis, new samples)
    or horizontal (same samples, extended spectral range).

    Parameters
    ----------
    dicts : list of dict
        Measurement-format dicts.
    how : str
        ``'auto'``, ``'vertical'``, or ``'horizontal'``.

    Returns
    -------
    str
        ``'vertical'`` or ``'horizontal'``.

    Raises
    ------
    ValueError
        If *how* is ``'auto'`` and the direction cannot be determined, or if
        the inputs are incompatible for the requested direction.
    """
    if how in ('vertical', 'horizontal'):
        return how

    # Identical xaxes → vertical merge
    try:
        _check_xaxis_compat([d['xaxis'] for d in dicts])
        return 'vertical'
    except ValueError:
        pass

    # Xaxes differ — verify they at least overlap (needed for horizontal)
    arrays = [np.asarray(d['xaxis']) for d in dicts]
    wn_lo = max(ax.min() for ax in arrays)
    wn_hi = min(ax.max() for ax in arrays)
    if wn_lo >= wn_hi:
        raise ValueError(
            "merge (auto): xaxis arrays neither match nor overlap.  "
            "Pass how='vertical' with resample=True to force merging on the "
            "intersection, or verify that the inputs are correct."
        )

    # Look for shared sample labels across any spectral sub-dict.  Blackbody
    # entries ('bbc'/'bbh') live alongside samples in 'rad'/'sbm' but are not
    # samples — excluding them prevents two vertically-stackable runs (disjoint
    # samples, shared BBs) from being misread as a horizontal merge.
    _bb_keys = {'bbc', 'bbh'}
    for key in ('emiss', 'rad', 'sbm', 'rad0'):
        label_sets = [
            set(d[key].keys()) - _bb_keys
            for d in dicts
            if isinstance(d.get(key), dict)
        ]
        if len(label_sets) >= 2 and label_sets[0] & label_sets[1]:
            return 'horizontal'

    raise ValueError(
        "merge (auto): xaxis arrays differ but no shared sample labels were "
        "found.  Use how='vertical' with resample=True to merge on the xaxis "
        "intersection, or how='horizontal' to force a spectral union merge."
    )


def _build_union_xaxis(xaxes: list) -> np.ndarray:
    """
    Build a wavenumber grid spanning the full union range of all inputs.

    Spacing is the finest (smallest) mean spacing found among the inputs so
    that no spectral detail is lost.  Direction (ascending / descending) is
    inherited from the first input.

    Parameters
    ----------
    xaxes : list of np.ndarray
        Wavenumber axes to span.

    Returns
    -------
    np.ndarray
        Uniform xaxis covering the union of all input ranges.
    """
    arrays    = [np.asarray(ax) for ax in xaxes]
    descending = arrays[0][0] > arrays[0][-1]
    wn_lo     = min(ax.min() for ax in arrays)
    wn_hi     = max(ax.max() for ax in arrays)
    spacing   = min(
        abs(float(ax[-1]) - float(ax[0])) / (len(ax) - 1)
        for ax in arrays
    )
    n_pts  = int(round((wn_hi - wn_lo) / spacing)) + 1
    union  = np.linspace(wn_lo, wn_hi, n_pts)
    logging.info(
        "merge (horizontal): union xaxis [%.2f, %.2f] cm⁻¹, %d pts @ %.4f cm⁻¹.",
        wn_lo, wn_hi, n_pts, spacing,
    )
    return union[::-1] if descending else union


def _overlap_dc_offset(
    xaxis1: np.ndarray,
    spec1:  np.ndarray,
    xaxis2: np.ndarray,
    spec2:  np.ndarray,
) -> float:
    """
    Compute the additive DC offset that aligns *spec2* to *spec1* in the
    overlap zone.

    The reference level is the maximum of *spec1* in the overlap.  *spec2* is
    shifted so that its own maximum in the overlap matches that value.  This
    corrects absolute instrument offsets under the assumption that the peak
    emissivity in the shared spectral region is a stable calibration anchor.

    Parameters
    ----------
    xaxis1, xaxis2 : np.ndarray
        Wavenumber axes of the two spectra (cm⁻¹).
    spec1, spec2 : np.ndarray
        Spectral arrays on their respective axes.

    Returns
    -------
    float
        Value to add to *spec2*: ``spec2_corrected = spec2 + offset``.
        Zero if the spectra do not overlap.
    """
    lo = max(xaxis1.min(), xaxis2.min())
    hi = min(xaxis1.max(), xaxis2.max())
    if lo >= hi:
        return 0.0
    mask1 = (xaxis1 >= lo) & (xaxis1 <= hi)
    mask2 = (xaxis2 >= lo) & (xaxis2 <= hi)
    if not mask1.any() or not mask2.any():
        return 0.0
    ref = float(np.nanmax(spec1[mask1]))
    return ref - float(np.nanmax(spec2[mask2]))


def _merge_spectral_union(dicts: list, align_overlap: bool = True) -> dict:
    """
    Horizontal measurement merge: same samples, extended spectral range.

    Constructs a union wavenumber grid spanning all inputs, resamples every
    spectral array onto it (NaN outside each instrument's own range), then
    combines with ``nanmean`` — averaging in the overlap, single-instrument
    values outside it.

    Parameters
    ----------
    dicts : list of dict
        Measurement-format dicts sharing sample labels.
    align_overlap : bool, optional
        If ``True`` (default), apply a per-label additive DC correction to
        each input beyond the first so its maximum in the overlap zone matches
        that of input 1.  Appropriate for emissivity; consider ``False`` for
        raw single-beam or radiance data.

    Returns
    -------
    dict
        Merged measurement dict on the union xaxis.  Non-spectral metadata
        (``calib``, scalars, ``label``, etc.) is taken from the first input.
    """
    _SPECTRAL_KEYS = ('emiss', 'rad', 'rad0', 'sbm')

    xaxes   = [np.asarray(d['xaxis']) for d in dicts]
    union_x = _build_union_xaxis(xaxes)

    # All unique sample labels across all spectral sub-dicts
    all_labels: set[str] = set()
    for d in dicts:
        for key in _SPECTRAL_KEYS:
            if isinstance(d.get(key), dict):
                all_labels.update(d[key].keys())

    merged: dict = {'xaxis': union_x, 'wl': 1e4 / union_x}

    for sub_key in _SPECTRAL_KEYS:
        if not any(isinstance(d.get(sub_key), dict) for d in dicts):
            continue
        merged_sub: dict = {}
        for label in sorted(all_labels):
            stacks: list[np.ndarray] = []
            ref_xaxis = ref_spec = None
            for i, d in enumerate(dicts):
                sub = d.get(sub_key)
                if not isinstance(sub, dict) or label not in sub:
                    continue
                src_xaxis = np.asarray(d['xaxis'])
                src_spec  = np.asarray(sub[label])
                if i == 0:
                    ref_xaxis, ref_spec = src_xaxis, src_spec
                elif align_overlap and ref_xaxis is not None:
                    offset = _overlap_dc_offset(ref_xaxis, ref_spec, src_xaxis, src_spec)
                    if offset != 0.0:
                        logging.info(
                            "merge (horizontal): DC offset %+.5f applied to "
                            "'%s'/'%s' from input %d.",
                            offset, sub_key, label, i + 1,
                        )
                    src_spec = src_spec + offset
                stacks.append(resample_spectrum(src_xaxis, src_spec, union_x))
            if stacks:
                merged_sub[label] = np.nanmean(np.vstack(stacks), axis=0)
        merged[sub_key] = merged_sub

    # Pass through non-spectral keys from the first input.  ``notes`` is
    # deliberately dropped: the embedded measurement-info table is row-aligned
    # to a single input's sample order and would be left inconsistent with the
    # merged union label set, so a horizontal merge carries no notes.
    _skip = {'xaxis', 'wl', 'notes'} | set(_SPECTRAL_KEYS)
    for key, val in dicts[0].items():
        if key not in _skip:
            merged[key] = val

    for i, d in enumerate(dicts[1:], start=2):
        for key in d:
            if key not in merged and key not in _skip:
                logging.warning(
                    "merge (horizontal): key '%s' in input %d absent from "
                    "input 1 — skipped.", key, i,
                )

    _sync_label_with_spectra(merged)
    return merged


def match(
    spec1: tuple,
    spec2: tuple,
    spacing: float | None = None,
    align_overlap: bool = True,
) -> dict:
    """
    Merge two single spectra covering overlapping but different spectral ranges.

    Convenience wrapper around the horizontal merge path for individual
    ``(xaxis, data)`` tuples rather than full measurement dicts.

    Parameters
    ----------
    spec1 : tuple of (np.ndarray, np.ndarray)
        ``(xaxis, data)`` for the first spectrum (used as the DC reference).
    spec2 : tuple of (np.ndarray, np.ndarray)
        ``(xaxis, data)`` for the second spectrum.
    spacing : float or None, optional
        Target xaxis spacing in cm⁻¹.  Defaults to the finest spacing of the
        two inputs so that no detail is lost.
    align_overlap : bool, optional
        If ``True`` (default), shift *spec2* so that its maximum in the overlap
        zone matches *spec1*'s maximum there.

    Returns
    -------
    dict
        ``{'xaxis': ..., 'wl': ..., 'data': ...}``
    """
    wn1, data1 = np.asarray(spec1[0]), np.asarray(spec1[1])
    wn2, data2 = np.asarray(spec2[0]), np.asarray(spec2[1])

    if spacing is None:
        spacing = min(
            abs(float(wn1[-1]) - float(wn1[0])) / (len(wn1) - 1),
            abs(float(wn2[-1]) - float(wn2[0])) / (len(wn2) - 1),
        )

    wn_lo = min(wn1.min(), wn2.min())
    wn_hi = max(wn1.max(), wn2.max())
    n_pts = int(round((wn_hi - wn_lo) / spacing)) + 1
    union_x = np.linspace(wn_lo, wn_hi, n_pts)
    if wn1[0] > wn1[-1]:
        union_x = union_x[::-1]

    if align_overlap:
        offset = _overlap_dc_offset(wn1, data1, wn2, data2)
        if offset != 0.0:
            logging.info("match: DC offset %+.5f applied to spec2.", offset)
        data2 = data2 + offset

    r1 = resample_spectrum(wn1, data1, union_x)
    r2 = resample_spectrum(wn2, data2, union_x)

    wn_overlap_lo = max(wn1.min(), wn2.min())
    wn_overlap_hi = min(wn1.max(), wn2.max())
    logging.info(
        "match: union [%.0f, %.0f] cm⁻¹ @ %.4f cm⁻¹; overlap [%.0f, %.0f] cm⁻¹.",
        wn_lo, wn_hi, spacing, wn_overlap_lo, wn_overlap_hi,
    )

    return {
        'xaxis': union_x,
        'wl':    1e4 / union_x,
        'data':  np.nanmean(np.vstack([r1, r2]), axis=0),
    }


# Datasets expected in a TES surface-emissivity result file.
_TES_KEYS = ('tesx73', 'surf_emiss73', 'det', 'ick', 'ock')


def is_tes_result(d: dict) -> bool:
    """
    Return ``True`` if *d* looks like a TES surface-emissivity result.

    Parameters
    ----------
    d : dict
        Dict as returned by :func:`utils.readHDF`.

    Returns
    -------
    bool
        ``True`` when every expected TES dataset name is present.
    """
    return all(k in d for k in _TES_KEYS)


def read_tes(path: str) -> dict:
    """
    Read a TES surface-emissivity HDF5 file into an emcal-style emissivity dict.

    Thermal Emission Spectrometer (TES) result files store surface emissivity
    retrievals under raw dataset names (``surf_emiss73`` / ``tesx73``) with
    per-spectrum detector and observation identifiers (``det`` / ``ick`` /
    ``ock``).  This converts them to the emissivity convention used throughout
    speclab (``xaxis`` / ``label`` / ``data`` / ``emiss``), so the result can be
    passed to :func:`merge`, saved via :func:`utils.saveHDF`, or loaded directly
    by the EmissionLWIR GUI.

    Spectra are labelled ``"OCK{ock}-ICK{ick}-det{det}"``, which uniquely
    identifies each TES pixel (orbit/observation counters + detector 1–6).

    Parameters
    ----------
    path : str
        Path to a TES result HDF5 file.

    Returns
    -------
    dict
        Emissivity result with keys ``xaxis`` (n_bands,), ``wl`` (n_bands,),
        ``label`` (list of n_spectra str), ``data`` (n_spectra, n_bands), and
        ``emiss`` ({label: spectrum}).

    Raises
    ------
    KeyError
        If the file lacks the expected TES datasets.
    ValueError
        If the emissivity band axis does not match the spectral axis length,
        or the per-spectrum identifiers do not align with the spectra count.
    """
    d = utils.readHDF(path)
    missing = [k for k in _TES_KEYS if k not in d]
    if missing:
        raise KeyError(
            f"{os.path.basename(path)} is not a TES result file "
            f"(missing {', '.join(missing)})."
        )

    xaxis = np.asarray(d['tesx73'], dtype=float).ravel()          # (n_bands,)
    emiss = np.asarray(d['surf_emiss73'], dtype=float)            # band axis first
    if emiss.ndim == 1:
        emiss = emiss[:, np.newaxis]
    if emiss.shape[0] != xaxis.size:
        raise ValueError(
            f"surf_emiss73 leading axis ({emiss.shape[0]}) does not match "
            f"tesx73 length ({xaxis.size}) in {os.path.basename(path)}."
        )
    data = emiss.reshape(emiss.shape[0], -1).T                    # (n_spectra, n_bands)

    det = np.atleast_1d(np.asarray(d['det'])).ravel()
    ick = np.atleast_1d(np.asarray(d['ick'])).ravel()
    ock = np.atleast_1d(np.asarray(d['ock'])).ravel()
    if not (len(det) == len(ick) == len(ock) == data.shape[0]):
        raise ValueError(
            f"TES identifier lengths (det={len(det)}, ick={len(ick)}, "
            f"ock={len(ock)}) do not match {data.shape[0]} spectra in "
            f"{os.path.basename(path)}."
        )
    labels = [f"OCK{int(o)}-ICK{int(i)}-det{int(x)}"
              for o, i, x in zip(ock, ick, det)]

    return {
        'xaxis': xaxis,
        'wl':    1e4 / xaxis,
        'label': labels,
        'data':  data,
        'emiss': {lbl: data[i] for i, lbl in enumerate(labels)},
    }


def merge(
    *inputs,
    how: str = 'auto',
    resample: bool = False,
    align_overlap: bool = True,
    save: bool = False,
    save_path: str | None = None,
) -> dict:
    """
    Merge two or more output dicts (or HDF5 file paths) into a single dict.

    Two structural formats are auto-detected:

    * **Per-entry (spectral library / album)** — all top-level values are dicts
      keyed by spectrum identifier.  Entries are combined; duplicate IDs with
      identical data are silently deduplicated, those with differing data are
      kept under a suffixed key.

    * **Measurement (emcal-style)** — top-level ``xaxis`` key present.  Two
      merge directions are available and auto-detected via *how*:

      * **Vertical** — same xaxis, different samples.  Sample arrays are
        concatenated and named-entry sub-dicts (``emiss``, ``rad``, …) are
        merged label-by-label.  Use ``resample=True`` when xaxes differ but
        the intersection is the desired output range.

      * **Horizontal** — same samples, different spectral regions that overlap.
        A union xaxis is built at the finest available spacing; both inputs are
        resampled onto it and combined with ``nanmean`` (averaging in the
        overlap, single-instrument values outside it).  An optional DC offset
        correction (``align_overlap``) aligns the absolute level of subsequent
        inputs to the first in the overlap zone.

    Parameters
    ----------
    *inputs : dict or str
        Two or more dicts, or path strings to HDF5 files.
    how : str, optional
        ``'auto'`` (default), ``'vertical'``, or ``'horizontal'``.  Auto-detection
        uses xaxis compatibility and shared sample labels as signals; pass
        explicitly when the inputs are ambiguous.
    resample : bool, optional
        Vertical merge only.  Build the common xaxis from the intersection of
        all input ranges before merging.  Default ``False``.
    align_overlap : bool, optional
        Horizontal merge only.  Apply a per-label DC offset to each input
        beyond the first so its maximum in the overlap zone matches that of
        input 1.  Default ``True``.
    save : bool, optional
        If ``True``, save the result via ``utils.saveHDF``.  Requires
        *save_path*.  Default ``False``.
    save_path : str or None, optional
        Output path for ``utils.saveHDF``.  Required when ``save=True``.

    Returns
    -------
    dict
        Merged dict in the same structural format as the inputs.

    Raises
    ------
    ValueError
        If fewer than two inputs are given, structural types are inconsistent,
        direction cannot be auto-detected, or ``save=True`` with no path.

    Examples
    --------
    Vertical merge of two emcal runs (same instrument, different samples):

    >>> combined = merge(out_samples_a, out_samples_b)

    Horizontal merge of MIR and FIR emcal runs (same samples, different ranges):

    >>> broadband = merge(out_mir, out_fir, how='horizontal')

    Vertical merge with incompatible xaxes (resampled to intersection):

    >>> combined = merge(out1, out2, resample=True)
    """
    if len(inputs) < 2:
        raise ValueError(f"merge requires at least 2 inputs; {len(inputs)} given.")
    if how not in ('auto', 'vertical', 'horizontal'):
        raise ValueError(
            f"merge: how must be 'auto', 'vertical', or 'horizontal'; got {how!r}."
        )

    # Load any file paths
    dicts = []
    for i, inp in enumerate(inputs, start=1):
        if isinstance(inp, str):
            logging.info("merge: loading input %d from '%s'", i, inp)
            dicts.append(utils.readHDF(inp))
        elif isinstance(inp, dict):
            dicts.append(inp)
        else:
            raise TypeError(
                f"merge: input {i} must be a dict or a file path str, "
                f"got {type(inp).__name__}."
            )

    # Detect and validate structural types
    types = [_detect_merge_type(d) for d in dicts]
    if len(set(types)) > 1:
        raise ValueError(
            f"merge: all inputs must share the same structural type; "
            f"found {set(types)}."
        )
    dtype = types[0]

    # Dispatch
    if dtype == 'per_entry':
        result = _merge_per_entry(dicts)
    else:
        direction = _detect_merge_direction(dicts, how)
        if direction == 'horizontal':
            result = _merge_spectral_union(dicts, align_overlap=align_overlap)
        else:
            result = _merge_measurement(dicts, resample=resample)

    if save:
        if save_path is None:
            raise ValueError("merge: save=True requires a save_path.")
        utils.saveHDF(result, save_path)
        logging.info("merge: saved merged dict to '%s'", save_path)

    return result


# =============================================================================
# =========================== tracal / refcal =================================
# =============================================================================

_RATIO_MIN_DENOM_FRACTION = 1e-3
"""Pixels where |denom| < this fraction of max(|denom|) are set to NaN."""


def _safe_divide(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    """
    Element-wise division with a near-zero denominator guard.

    Any pixel where ``|denom|`` is below ``_RATIO_MIN_DENOM_FRACTION`` of the
    denominator's own peak absolute value is replaced with NaN rather than
    producing ±Inf.  The numerator is not thresholded.

    Parameters
    ----------
    num, denom : np.ndarray
        Arrays of the same shape.

    Returns
    -------
    np.ndarray
        ``num / denom`` with near-zero denominator pixels set to ``np.nan``.
    """
    peak = np.nanmax(np.abs(denom))
    if peak == 0.0:
        return np.full_like(num, np.nan, dtype=float)
    threshold = peak * _RATIO_MIN_DENOM_FRACTION
    safe_denom = np.where(np.abs(denom) >= threshold, denom, np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        return num / safe_denom


def _ratio_cal_impl(
    fdir: str,
    ext: str,
    notes_path: 'str | None',
    plot: bool,
    save: bool,
    ow: bool,
    mode: str,
) -> dict:
    """
    Shared implementation for :func:`tracal` and :func:`refcal`.

    Parameters
    ----------
    mode : str
        ``'transmittance'`` or ``'reflectance'``.  Controls output key names,
        saved file prefix, and whether absorbance / optical depth are computed.
    """
    if '~' in fdir:
        fdir = os.path.expanduser(fdir)
    fdir = os.path.abspath(fdir)

    # ── 1. Load measurement-info CSV ─────────────────────────────────────────
    df, flist = _load_notes(notes_path if notes_path is not None else fdir)
    if df is None:
        raise IOError(
            "No measurement-info file found in %s. "
            "Collect spectra with AutomateFTIR first." % fdir
        )

    required_cols = {'sample_name', 'is_bkg', 'is_blank', 'is_bb', 'dtime'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            "Measurement-info file is missing columns: %s. "
            "Is this an AutomateFTIR notes file?" % sorted(missing)
        )

    df = df.copy()
    df['_dtime'] = pd.to_datetime(df['dtime'], format='mixed', errors='coerce')

    # ── 2. Partition rows by spectrum type ────────────────────────────────────
    is_bkg_col   = df['is_bkg'].astype(int)
    is_blank_col = df['is_blank'].astype(int)
    is_bb_col    = df['is_bb'].astype(int)

    bkg_rows    = df[is_bkg_col   == 1]
    blank_rows  = df[is_blank_col == 1]
    sample_rows = df[(is_bb_col == 0) & (is_bkg_col == 0) & (is_blank_col == 0)]

    if bkg_rows.empty:
        raise IOError("No background spectra (is_bkg=1) in measurement-info file.")
    if sample_rows.empty:
        raise IOError("No sample spectra in measurement-info file.")

    has_blank = not blank_rows.empty

    # ── 3. File loader ────────────────────────────────────────────────────────
    ext_upper = ext.upper()
    ext_lower = ext.lower()

    def _load_spectrum(sample_name: str) -> dict:
        for suffix in (ext_upper, ext_lower):
            fpath = os.path.join(fdir, sample_name + suffix)
            if os.path.exists(fpath):
                return utils.readOMNIC(fpath)
        raise IOError(
            "Spectrum file not found for '%s' (tried %s, %s) in %s"
            % (sample_name, ext_upper, ext_lower, fdir)
        )

    # ── 4. Load reference spectra (bkg and blank) ─────────────────────────────
    # Cache as {name: (spec_dict, timestamp)} — first occurrence wins.
    bkg_cache: dict = {}
    blank_cache: dict = {}

    for _, row in bkg_rows.iterrows():
        name = row['sample_name']
        if name not in bkg_cache:
            try:
                bkg_cache[name] = (_load_spectrum(name), row['_dtime'])
            except IOError as exc:
                logging.warning("Skipping bkg '%s': %s", name, exc)

    if has_blank:
        for _, row in blank_rows.iterrows():
            name = row['sample_name']
            if name not in blank_cache:
                try:
                    blank_cache[name] = (_load_spectrum(name), row['_dtime'])
                except IOError as exc:
                    logging.warning("Skipping blank '%s': %s", name, exc)
        if not blank_cache:
            logging.warning(
                "All blank files failed to load — falling back to bkg directly."
            )
            has_blank = False

    if not bkg_cache:
        raise IOError("All background files failed to load from %s." % fdir)

    # ── 5. Closest-in-time helper ─────────────────────────────────────────────
    def _closest(cache: dict, t_sample: 'pd.Timestamp') -> dict:
        """Return the spectrum dict whose timestamp is closest to *t_sample*."""
        best = min(
            cache.items(),
            key=lambda kv: abs((kv[1][1] - t_sample).total_seconds()),
        )
        return best[1][0]   # (name, (spec, t)) → spec

    # Wavenumber axis from first background
    wn = next(iter(bkg_cache.values()))[0]['wn']

    # ── 6. Compute ratios per sample ──────────────────────────────────────────
    labels     = []
    ratio_arr  = []
    sbm_out    = {}

    for _, row in sample_rows.iterrows():
        name = row['sample_name']
        try:
            spec = _load_spectrum(name)
        except IOError as exc:
            logging.warning("Skipping sample '%s': %s", name, exc)
            continue

        t_sample = row['_dtime']
        bkg      = _closest(bkg_cache, t_sample)

        if has_blank:
            blank = _closest(blank_cache, t_sample)
            # (sample/bkg) / (blank/bkg)  =  sample / blank
            ratio = _safe_divide(spec['data'], blank['data'])
        else:
            ratio = _safe_divide(spec['data'], bkg['data'])

        labels.append(name)
        ratio_arr.append(ratio)
        sbm_out[name] = spec['data']

    if not labels:
        raise IOError("No sample spectra could be loaded from %s." % fdir)

    ratio_matrix = np.vstack(ratio_arr)
    ratio_key    = 'tra' if mode == 'transmittance' else 'ref'

    # ── 7. Build output dict ──────────────────────────────────────────────────
    out: dict = {}
    out['header'] = {
        'Processed':     datetime.now().isoformat(),
        'mode':          mode,
        'bkg_names':     list(bkg_cache.keys()),
        'blank_names':   list(blank_cache.keys()),
        'use_blank':     has_blank,
        'sample_labels': labels,
        'notes_file':    flist[0] if flist else '',
    }
    out['wn'] = wn
    out['wl'] = 1e4 / wn

    out['sbm'] = {'wn': wn}
    for bkg_name, (bkg_spec, _) in bkg_cache.items():
        out['sbm'][f'bkg:{bkg_name}'] = bkg_spec['data']
    if has_blank:
        for bl_name, (bl_spec, _) in blank_cache.items():
            out['sbm'][f'blank:{bl_name}'] = bl_spec['data']
    out['sbm'].update(sbm_out)

    out[ratio_key] = {'wn': wn}
    for i, name in enumerate(labels):
        out[ratio_key][name] = ratio_matrix[i]
    out[ratio_key]['mean'] = np.nanmean(ratio_matrix, axis=0)
    out[ratio_key]['std']  = np.nanstd(ratio_matrix,  axis=0)

    if mode == 'transmittance':
        with np.errstate(invalid='ignore', divide='ignore'):
            abs_matrix = -np.log10(ratio_matrix)
            od_matrix  = -np.log(ratio_matrix)

        out['abs'] = {'wn': wn}
        for i, name in enumerate(labels):
            out['abs'][name] = abs_matrix[i]
        out['abs']['mean'] = np.nanmean(abs_matrix, axis=0)
        out['abs']['std']  = np.nanstd(abs_matrix,  axis=0)

        out['od'] = {'wn': wn}
        for i, name in enumerate(labels):
            out['od'][name] = od_matrix[i]
        out['od']['mean'] = np.nanmean(od_matrix, axis=0)
        out['od']['std']  = np.nanstd(od_matrix,  axis=0)

    # ── 8. Save results ───────────────────────────────────────────────────────
    if save:
        prefix    = 'tracal'   if mode == 'transmittance' else 'refcal'
        timestamp = '' if ow else datetime.now().strftime('%Y%m%d_%H%M%S')
        sep       = '_' if timestamp else ''

        hdf_path = os.path.join(fdir, f'{prefix}_results{sep}{timestamp}.hdf')
        utils.saveHDF(out, hdf_path)
        logging.info("%s HDF saved: %s", mode, hdf_path)

        csv_path = os.path.join(fdir, f'{prefix}_results{sep}{timestamp}.csv')
        if mode == 'transmittance':
            utils.save_tracal_csv(out, csv_path)
        else:
            utils.save_refcal_csv(out, csv_path)
        logging.info("%s CSV saved: %s", mode, csv_path)

    return out


def tracal(
    fdir: str,
    ext: str = '.csv',
    notes_path: 'str | None' = None,
    plot: bool = False,
    save: bool = False,
    ow: bool = False,
) -> dict:
    """
    Perform transmission calibration on an AutomateFTIR measurement folder.

    Uses the ``*-measurement-info.csv`` metadata file to identify background
    (``is_bkg=1``), blank (``is_blank=1``), and sample spectra, then pairs
    each sample with its **closest-in-time** background and blank.

    Formulae
    --------
    Without blank:  ``T = sample / bkg``
    With blank:     ``T = (sample / bkg) / (blank / bkg) = sample / blank``
    Absorbance:     ``A = −log₁₀(T)``
    Optical depth:  ``OD = −ln(T)``

    Parameters
    ----------
    fdir : str
        Path to the measurement folder.  ``~`` is expanded.
    ext : str
        Spectral file extension (default ``'.csv'``).
    notes_path : str or None
        Explicit path to the measurement-info CSV.  Auto-located if ``None``.
    plot : bool
        Not yet implemented (reserved for a future summary plot).
    save : bool
        If True, write results to ``tracal_results_<timestamp>.hdf`` and
        ``.csv`` inside *fdir*.
    ow : bool
        If True, omit the timestamp from saved filenames (overwrite mode).

    Returns
    -------
    dict
        Keys: ``header``, ``wn``, ``wl``, ``sbm``, ``tra``, ``abs``, ``od``.
        Each of ``tra``, ``abs``, ``od`` contains per-sample arrays plus
        ``mean`` and ``std``.
    """
    return _ratio_cal_impl(
        fdir, ext=ext, notes_path=notes_path,
        plot=plot, save=save, ow=ow, mode='transmittance',
    )


def refcal(
    fdir: str,
    ext: str = '.csv',
    notes_path: 'str | None' = None,
    plot: bool = False,
    save: bool = False,
    ow: bool = False,
) -> dict:
    """
    Perform reflectance calibration on an AutomateFTIR measurement folder.

    Identical logic to :func:`tracal` but uses ``'reflectance'`` semantics:
    output key is ``'ref'`` instead of ``'tra'``; absorbance and optical depth
    are not computed.

    Parameters
    ----------
    fdir : str
        Path to the measurement folder.  ``~`` is expanded.
    ext : str
        Spectral file extension (default ``'.csv'``).
    notes_path : str or None
        Explicit path to the measurement-info CSV.  Auto-located if ``None``.
    plot : bool
        Not yet implemented (reserved for a future summary plot).
    save : bool
        If True, write results to ``refcal_results_<timestamp>.hdf`` and
        ``.csv`` inside *fdir*.
    ow : bool
        If True, omit the timestamp from saved filenames (overwrite mode).

    Returns
    -------
    dict
        Keys: ``header``, ``wn``, ``wl``, ``sbm``, ``ref``.
        ``ref`` contains per-sample arrays plus ``mean`` and ``std``.
    """
    return _ratio_cal_impl(
        fdir, ext=ext, notes_path=notes_path,
        plot=plot, save=save, ow=ow, mode='reflectance',
    )

