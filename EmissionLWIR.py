#!/usr/bin/env python3
"""
EmissionLWIR — GUI for laboratory spectroscopy data processing.

Toolbar
-------Does
Load Data | Load Library | Build Library | Run emcal() | Run sma()

Tabs
----
Data     : radiance / emissivity spectra + measurement info table
Speclib  : spectral library browser with excluded / forced endmember designation
Analysis : SMA results — individual fits, concentrations, pie charts
"""

import os
import logging
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import cmcrameri

# Package bootstrap so the file is runnable directly as well as via entry point.
if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'speclab'

from .functions import load_sbm, cal_rad, emcal, sma, resample_spectrum, merge, scan_sample_labels, MissingTempsError
from .plot import _add_top_axis
from .utils import readOMNIC, readEmissionCSVnotes, findFiles, readHDF, saveHDF, save_emcal_csv, save_sma_csv, c2k, r2t_nau, _to_album, _set_window_size
from .config import get_config
from . import __version__
from .SpeclibViewer import (
    SpeclibViewer,
    _load_hdf,
    _INFO_GROUPS,
    _INFO_COLLAPSED_BY_DEFAULT,
)

# ---------------------------------------------------------------------------
# Defaults and constants
# ---------------------------------------------------------------------------

_EMCAL_DEFAULTS: dict = dict(
    lab='nau', method='nem', max_emiss=1.0, bb_emiss=0.995,
    n_bb=2, temp_halfwidth=50.0, violation_weight=5.0, violation_tol=0.0,
    escalation_factor=4.0, max_escalations=4,
    noise_free=True, apply_dehyd=False,
    wn_range=(500.0, 1700.0),
)

_CALRAD_DEFAULTS: dict = dict(
    lab='nau', bb_emiss=0.995, noise_free=True,
)

_SMA_DEFAULTS: dict = dict(
    wn_range=(300.0, 1800.0), bb=True, group=True, nn=True,
    calc_errors=True, notchco2=0.0, slope=False, sample_t=None,
)

_TAG_EXCL  = 'excluded'
_TAG_FORCE = 'forced'
_PAGE_SIZE = 10

# ── Per-sample key registries (used by the "Delete spectrum" feature) ────────
# Only keys whose leading axis (or dict membership) indexes the sample are
# listed here; everything else — band axes, endmember-axis arrays, calib,
# scalars — is shared and must be left untouched when a sample is removed.

# emcal result: label-keyed sub-dicts. ``rad`` / ``sbm`` additionally hold
# 'bbc' / 'bbh' entries, which are removed by label (never matched) → safe.
_EMCAL_SAMPLE_DICTS = (
    'emiss', 'emiss_full', 'rad0', 'sample_temps',
    'sample_t_wavenumber', 'rad', 'sbm',
)

# sma result: top-level arrays/lists whose axis-0 is the sample index.
_SMA_SAMPLE_KEYS = (
    'sample_labels', 'conc', 'normconc', 'bb', 'bb_normconc', 'rms',
    'error', 'bberror', 'slope', 'slopeerror', 'normerror', 'slope_normconc',
    'delta_t_estimated', 'measured', 'measured_fit', 'modeled', 'sort',
    'atm_conc',
)

# sma result: per-sample arrays nested inside the ``grouped`` sub-dict.
_SMA_GROUPED_SAMPLE_KEYS = (
    'grouped_bb', 'grouped_bberror', 'grouped_conc', 'grouped_error',
    'grouped_normconc', 'grouped_normerror', 'grouped_slope',
    'grouped_slopeerror', 'grouped_sort',
)


def _delete_axis0(value: object, idx: int, n: int) -> object:
    """
    Remove index *idx* along the leading axis of *value* iff its length is *n*.

    Handles NumPy arrays and Python lists; any other type, or a length that
    does not match *n*, is returned unchanged. The length guard is what keeps
    shared axes (band grids, endmember arrays) and mixed-length sub-tables
    (e.g. length-2 blackbody note columns) from being touched.

    Parameters
    ----------
    value : object
        Candidate per-sample container.
    idx : int
        Sample index to drop.
    n : int
        Expected number of samples; only containers of this leading length
        are modified.

    Returns
    -------
    object
        A new container with the sample removed, or *value* unchanged.
    """
    if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n:
        return np.delete(value, idx, axis=0)
    if isinstance(value, list) and len(value) == n:
        return [x for i, x in enumerate(value) if i != idx]
    return value

# Named color palettes available via the toolbar color-scheme selector.
# 'Standard' and 'Dark' are matplotlib qualitative; the rest are
def _lighten_colors(colors: list, factor: float = 0.5) -> list:
    """Return a copy of *colors* with each HLS lightness blended toward 1.0 by *factor*."""
    import colorsys
    out = []
    for r, g, b, a in colors:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l2 = l + (1.0 - l) * factor
        r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
        out.append((r2, g2, b2, a))
    return out


def _cmc_colors(name: str) -> list | None:
    """Return the discrete color list for a cmcrameri S colormap, or None if absent."""
    try:
        return list(getattr(cmcrameri.cm, name).colors)
    except AttributeError:
        return None


_dark2       = [plt.cm.Dark2(i) for i in range(8)]
_tab20_dark  = [plt.cm.tab20(i) for i in range(0, 20, 2)]
_tab20_light = [plt.cm.tab20(i) for i in range(1, 20, 2)]

_COLOR_SCHEMES: dict[str, list] = {
    'Standard': _tab20_dark + _tab20_light,            # 20 — tab20 dark+light hues
    'Dark':     _dark2 + _lighten_colors(_dark2),      # 16 — high-contrast Dark2
}

# Four cmcrameri S maps selected for maximum hue spread / visual separation.
# All have 100 discrete colors; gracefully absent if the package is old.
for _cmc_name in ['batlowS', 'hawaiiS', 'lipariS', 'tokyoS']:
    _cols = _cmc_colors(_cmc_name)
    if _cols is not None:
        _COLOR_SCHEMES[_cmc_name] = _cols
_COLOR_SCHEME_DEFAULT = 'Standard'
_AN_COLOR_OTHER = '#cccccc'

_CHANNEL_LABELS: dict[int, str] = {
    101: 'BB resistance low',
    102: 'BB resistance high',
    103: 'Mirror',
    104: 'Chamber exterior',
    105: 'Chamber interior',
    106: 'Chamber door',
    107: 'Detector',
}

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=_COLOR_SCHEMES[_COLOR_SCHEME_DEFAULT])

plt.rcParams.update({
    'font.size':       11,
    'lines.linewidth': 1.0,
    'xtick.direction': 'in', 'xtick.top':   True,
    'ytick.direction': 'in', 'ytick.right': True,
})
# prop_cycle is set after _COLOR_SCHEMES is defined below

# ---------------------------------------------------------------------------
# Modal option dialogs
# ---------------------------------------------------------------------------

class BBTempsDialog(tk.Toplevel):
    """
    Modal popup for entering warm and hot blackbody temperatures.

    Each field accepts either a temperature in °C (single value) or a
    resistance pair ``ch1 ch2`` or ``ch1, ch2`` (converted via r2t_nau).
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title('Blackbody Temperatures')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: tuple[float, float] | None = None  # (bb1_K, bb2_K)
        self._build()
        self.wait_window(self)

    @staticmethod
    def _parse(raw: str, field: str) -> float:
        parts = raw.replace(',', ' ').split()
        if len(parts) == 1:
            return c2k(float(parts[0]))
        elif len(parts) == 2:
            return r2t_nau(float(parts[0]), float(parts[1]))
        raise ValueError(f"{field}: expected a temperature (°C) or resistance pair (ch1 ch2)")

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text='No measurement info file found.\n'
                 'Enter blackbody temperatures.',
            foreground='#c05000',
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        ttk.Label(
            frm,
            text='Enter a temperature in °C or a resistance pair as "ch1 ch2".',
            foreground='#666666',
            justify=tk.LEFT,
            wraplength=340,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        ttk.Label(frm, text='Warm BB:').grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._bb1_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._bb1_var, width=18).grid(
            row=2, column=1, sticky=tk.W)

        ttk.Label(frm, text='Hot BB:').grid(
            row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._bb2_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._bb2_var, width=18).grid(
            row=3, column=1, sticky=tk.W)

        bf = ttk.Frame(frm)
        bf.grid(row=4, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(bf, text='OK',     command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _ok(self) -> None:
        try:
            bb1_k = self._parse(self._bb1_var.get().strip(), 'Warm BB')
            bb2_k = self._parse(self._bb2_var.get().strip(), 'Hot BB')
            self.result = (bb1_k, bb2_k)
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


class DownwellingTempsDialog(tk.Toplevel):
    """Modal popup for manually entering per-sample downwelling temperatures."""

    def __init__(self, master: tk.Misc, labels: list[str],
                 folder_name: str = '') -> None:
        super().__init__(master)
        title = f'Downwelling Temperatures — {folder_name}' if folder_name else 'Downwelling Temperatures'
        self.title(title)
        self.resizable(False, True)
        self.grab_set()
        self.transient(master)
        self.result: dict[str, float] | None = None
        self._labels = labels
        self._folder_name = folder_name
        self._entry_vars: dict[str, tk.StringVar] = {}
        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=14)
        outer.pack(fill=tk.BOTH, expand=True)

        msg = 'No measurement info file found.'
        if self._folder_name:
            msg += f'\nFolder: {self._folder_name}'
        msg += '\nEnter the downwelling (ambient) temperature in °C for each sample.'
        ttk.Label(
            outer,
            text=msg,
            foreground='#c05000',
            justify=tk.LEFT,
            wraplength=360,
        ).pack(anchor=tk.W, pady=(0, 10))

        # "Apply to all" shortcut
        shortcut_frm = ttk.Frame(outer)
        shortcut_frm.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(shortcut_frm, text='Apply to all:').pack(side=tk.LEFT, padx=(0, 6))
        self._all_var = tk.StringVar()
        ttk.Entry(shortcut_frm, textvariable=self._all_var, width=9).pack(side=tk.LEFT)
        ttk.Button(shortcut_frm, text='Apply', command=self._apply_all).pack(
            side=tk.LEFT, padx=(4, 0))

        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # Scrollable sample grid
        container = ttk.Frame(outer)
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        def _on_inner_configure(event):
            canvas.configure(scrollregion=canvas.bbox('all'))

        def _on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)

        inner.bind('<Configure>', _on_inner_configure)
        canvas.bind('<Configure>', _on_canvas_configure)

        hdr_font = ('TkDefaultFont', 9, 'bold')
        ttk.Label(inner, text='Sample', font=hdr_font).grid(
            row=0, column=0, sticky=tk.W, padx=(0, 16), pady=(0, 4))
        ttk.Label(inner, text='Temp (°C)', font=hdr_font).grid(
            row=0, column=1, sticky=tk.W, pady=(0, 4))

        for i, label in enumerate(self._labels, start=1):
            ttk.Label(inner, text=label).grid(
                row=i, column=0, sticky=tk.W, padx=(0, 16), pady=2)
            var = tk.StringVar()
            ttk.Entry(inner, textvariable=var, width=10).grid(
                row=i, column=1, sticky=tk.W, pady=2)
            self._entry_vars[label] = var

        row_h = 26
        visible_rows = min(len(self._labels) + 1, 12)
        canvas.configure(height=visible_rows * row_h + 8, width=360)

        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        bf = ttk.Frame(outer)
        bf.pack()
        ttk.Button(bf, text='OK',     command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _apply_all(self) -> None:
        val = self._all_var.get().strip()
        for var in self._entry_vars.values():
            var.set(val)

    def _ok(self) -> None:
        try:
            result: dict[str, float] = {}
            for label, var in self._entry_vars.items():
                s = var.get().strip()
                if not s:
                    raise ValueError(f"Temperature for '{label}' is empty.")
                result[label] = float(s)
            self.result = result
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


class EmcalOptionsDialog(tk.Toplevel):
    """Modal popup collecting emcal() keyword arguments."""

    def __init__(self, master: tk.Misc, defaults: dict) -> None:
        super().__init__(master)
        self.title('emcal() Options')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: dict | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._build(defaults)
        self.wait_window(self)

    def _build(self, d: dict) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        rows = [
            ('lab',              'Lab',                       'combo', ['nau', 'asu', 'swri', 'spectrometer']),
            ('method',           'Method',                    'combo', ['nem', 'alpha', 'hullfit_linear', 'hullfit', 'mmd']),
            ('max_emiss',        'Max emissivity',            'float', None),
            ('bb_emiss',         'BB emissivity',             'float', None),
            ('n_bb',             'N BB (hullfit)',            'int',   None),
            ('temp_halfwidth',   'Temp half-width K (hullfit)', 'float', None),
            ('violation_weight', 'Violation weight (hullfit)', 'float', None),
            ('violation_tol',    'Violation tol (hullfit)',   'float', None),
            ('escalation_factor','Escalation factor (hullfit)','float', None),
            ('max_escalations',  'Max escalations (hullfit)', 'int',   None),
            ('noise_free',       'Noise-free IRF',            'bool',  None),
            ('apply_dehyd',      'Apply dehyd',               'bool',  None),
        ]
        for row, (key, label, kind, opts) in enumerate(rows):
            ttk.Label(frm, text=f'{label}:').grid(
                row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
            val = d.get(key)
            if kind == 'bool':
                var = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(frm, variable=var).grid(row=row, column=1, sticky=tk.W)
            elif kind == 'combo':
                var = tk.StringVar(value=str(val))
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state='readonly', width=18).grid(row=row, column=1, sticky=tk.W)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(frm, textvariable=var, width=12).grid(row=row, column=1, sticky=tk.W)
            self._vars[key] = var

        wn_lo, wn_hi = d.get('wn_range', (500.0, 1700.0))
        ttk.Label(frm, text='Wn range (cm⁻¹):').grid(
            row=len(rows), column=0, sticky=tk.W, padx=(0, 10), pady=3)
        rng_frm = ttk.Frame(frm)
        rng_frm.grid(row=len(rows), column=1, sticky=tk.W)
        var_wn_lo = tk.StringVar(value=str(int(wn_lo)))
        var_wn_hi = tk.StringVar(value=str(int(wn_hi)))
        ttk.Entry(rng_frm, textvariable=var_wn_lo, width=7).pack(side=tk.LEFT)
        ttk.Label(rng_frm, text=' – ').pack(side=tk.LEFT)
        ttk.Entry(rng_frm, textvariable=var_wn_hi, width=7).pack(side=tk.LEFT)
        self._vars['wn_lo'] = var_wn_lo
        self._vars['wn_hi'] = var_wn_hi

        bf = ttk.Frame(frm)
        bf.grid(row=len(rows) + 1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(bf, text='Run',    command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _ok(self) -> None:
        try:
            self.result = {
                'lab':               self._vars['lab'].get(),
                'method':            self._vars['method'].get(),
                'max_emiss':         float(self._vars['max_emiss'].get()),
                'bb_emiss':          float(self._vars['bb_emiss'].get()),
                'n_bb':              int(self._vars['n_bb'].get()),
                'temp_halfwidth':    float(self._vars['temp_halfwidth'].get()),
                'violation_weight':  float(self._vars['violation_weight'].get()),
                'violation_tol':     float(self._vars['violation_tol'].get()),
                'escalation_factor': float(self._vars['escalation_factor'].get()),
                'max_escalations':   int(self._vars['max_escalations'].get()),
                'noise_free':        bool(self._vars['noise_free'].get()),
                'apply_dehyd':       bool(self._vars['apply_dehyd'].get()),
                'wn_range':          (float(self._vars['wn_lo'].get()),
                                      float(self._vars['wn_hi'].get())),
            }
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


class CalRadOptionsDialog(tk.Toplevel):
    """Modal popup collecting cal_rad() keyword arguments."""

    def __init__(self, master: tk.Misc, defaults: dict) -> None:
        super().__init__(master)
        self.title('cal_rad() Options')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: dict | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._build(defaults)
        self.wait_window(self)

    def _build(self, d: dict) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        rows = [
            ('lab',        'Lab',             'combo', ['nau', 'asu', 'swri', 'spectrometer']),
            ('bb_emiss',   'BB emissivity',   'float', None),
            ('noise_free', 'Noise-free IRF',  'bool',  None),
        ]
        for row, (key, label, kind, opts) in enumerate(rows):
            ttk.Label(frm, text=f'{label}:').grid(
                row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
            val = d.get(key)
            if kind == 'bool':
                var = tk.BooleanVar(value=bool(val))
                ttk.Checkbutton(frm, variable=var).grid(row=row, column=1, sticky=tk.W)
            elif kind == 'combo':
                var = tk.StringVar(value=str(val))
                ttk.Combobox(frm, textvariable=var, values=opts,
                             state='readonly', width=18).grid(row=row, column=1, sticky=tk.W)
            else:
                var = tk.StringVar(value=str(val))
                ttk.Entry(frm, textvariable=var, width=12).grid(row=row, column=1, sticky=tk.W)
            self._vars[key] = var

        bf = ttk.Frame(frm)
        bf.grid(row=len(rows), column=0, columnspan=2, pady=(12, 0))
        ttk.Button(bf, text='Run',    command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _ok(self) -> None:
        try:
            self.result = {
                'lab':        self._vars['lab'].get(),
                'bb_emiss':   float(self._vars['bb_emiss'].get()),
                'noise_free': bool(self._vars['noise_free'].get()),
            }
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


class SmaOptionsDialog(tk.Toplevel):
    """Modal popup collecting sma() keyword arguments."""

    def __init__(self, master: tk.Misc, defaults: dict) -> None:
        super().__init__(master)
        self.title('sma() Options')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: dict | None = None
        self._vars: dict[str, tk.Variable] = {}
        self._build(defaults)
        self.wait_window(self)

    def _build(self, d: dict) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        row = 0

        # Wn range — inline entry pair
        wn_lo, wn_hi = d.get('wn_range', (500.0, 1700.0))
        ttk.Label(frm, text='Wn range (cm⁻¹):').grid(
            row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
        rng_frm = ttk.Frame(frm)
        rng_frm.grid(row=row, column=1, sticky=tk.W)
        var_wn_lo = tk.StringVar(value=str(int(wn_lo)))
        var_wn_hi = tk.StringVar(value=str(int(wn_hi)))
        ttk.Entry(rng_frm, textvariable=var_wn_lo, width=7).pack(side=tk.LEFT)
        ttk.Label(rng_frm, text=' – ').pack(side=tk.LEFT)
        ttk.Entry(rng_frm, textvariable=var_wn_hi, width=7).pack(side=tk.LEFT)
        self._vars['wn_lo'] = var_wn_lo
        self._vars['wn_hi'] = var_wn_hi
        row += 1

        ttk.Label(frm, text='CO₂ notch (cm⁻¹):').grid(
            row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
        notch_var = tk.StringVar(value=str(d.get('notchco2', 0.0)))
        ttk.Entry(frm, textvariable=notch_var, width=12).grid(row=row, column=1, sticky=tk.W)
        self._vars['notchco2'] = notch_var
        row += 1

        bool_rows = [
            ('bb',           'Include blackbody',  d.get('bb',    True)),
            ('group',        'Group by type',      d.get('group', True)),
            ('nn',           'Non-negative',        d.get('nn',    True)),
            ('calc_errors',  'Calc. errors',        d.get('calc_errors', True)),
            ('slope',        'Slope endmember',     d.get('slope', False)),
        ]
        for key, label, val in bool_rows:
            ttk.Label(frm, text=f'{label}:').grid(
                row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
            var = tk.BooleanVar(value=bool(val))
            ttk.Checkbutton(frm, variable=var).grid(row=row, column=1, sticky=tk.W)
            self._vars[key] = var
            row += 1

        # Sample T — single entry, empty means auto from emcal
        st_val = d.get('sample_t')
        ttk.Label(frm, text='Sample T (K):').grid(
            row=row, column=0, sticky=tk.W, padx=(0, 10), pady=3)
        var_st = tk.StringVar(value='' if st_val is None else str(st_val))
        e = ttk.Entry(frm, textvariable=var_st, width=12)
        e.grid(row=row, column=1, sticky=tk.W)
        e.insert(0, '')
        # placeholder-style hint in the label
        ttk.Label(frm, text='(blank = auto from emcal)',
                  foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(4, 0))
        self._vars['sample_t'] = var_st
        row += 1

        bf = ttk.Frame(frm)
        bf.grid(row=row, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(bf, text='Run',    command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

    def _ok(self) -> None:
        try:
            self.result = {
                'wn_range':    (float(self._vars['wn_lo'].get()),
                                float(self._vars['wn_hi'].get())),
                'notchco2':    float(self._vars['notchco2'].get()),
                'bb':          bool(self._vars['bb'].get()),
                'group':       bool(self._vars['group'].get()),
                'nn':          bool(self._vars['nn'].get()),
                'calc_errors': bool(self._vars['calc_errors'].get()),
                'slope':    bool(self._vars['slope'].get()),
                'sample_t': (None if not self._vars['sample_t'].get().strip()
                             else float(self._vars['sample_t'].get())),
            }
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


# ---------------------------------------------------------------------------
# Folder selection dialog (multi-folder emcal mode)
# ---------------------------------------------------------------------------

class FolderSelectDialog(tk.Toplevel):
    """Modal dialog for selecting subfolders within a parent directory."""

    def __init__(self, master: tk.Misc, parent_dir: str,
                 preselected: list[str] | None = None) -> None:
        super().__init__(master)
        self.title('Select Folders')
        self.resizable(True, True)
        self.grab_set()
        self.transient(master)
        self.result: list[str] | None = None   # None = cancelled

        self._parent_dir = parent_dir
        subdirs = sorted(
            f.path for f in os.scandir(parent_dir) if f.is_dir()
        )
        self._subdirs = subdirs
        self._build(subdirs, preselected or subdirs)
        self.wait_window(self)

    def _build(self, subdirs: list[str], preselected: list[str]) -> None:
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text=f'Parent: {self._parent_dir}',
                  foreground='gray').pack(anchor=tk.W, pady=(0, 6))

        lf = ttk.LabelFrame(frm, text='Subfolders', padding=4)
        lf.pack(fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self._lb = tk.Listbox(lf, selectmode=tk.MULTIPLE, exportselection=False,
                              yscrollcommand=sb.set, font=('TkFixedFont', 11),
                              activestyle='dotbox')
        sb.config(command=self._lb.yview)
        self._lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        presel_set = set(preselected)
        for i, path in enumerate(subdirs):
            self._lb.insert(tk.END, Path(path).name)
            if path in presel_set:
                self._lb.selection_set(i)

        # Select all / none helpers
        sel_row = ttk.Frame(frm)
        sel_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(sel_row, text='Select all',
                   command=lambda: self._lb.selection_set(0, tk.END)).pack(side=tk.LEFT, padx=2)
        ttk.Button(sel_row, text='Deselect all',
                   command=lambda: self._lb.selection_clear(0, tk.END)).pack(side=tk.LEFT, padx=2)

        bf = ttk.Frame(frm)
        bf.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(bf, text='OK',     command=self._ok).pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=4)

        self.geometry('420x380')

    def _ok(self) -> None:
        selected = [self._subdirs[i] for i in self._lb.curselection()]
        if not selected:
            messagebox.showwarning('No selection',
                                   'Select at least one folder.', parent=self)
            return
        self.result = selected
        self.destroy()


# ---------------------------------------------------------------------------
# Axis Limits dialog
# ---------------------------------------------------------------------------

class AxisLimitsDialog(tk.Toplevel):
    """
    Persistent (non-modal) dialog for manually setting axis limits on the
    SMA analysis plot.  Stays open while the user navigates between samples;
    limits are re-applied after every redraw.
    """

    def __init__(self, viewer: 'EmissionLWIR') -> None:
        super().__init__(viewer)
        self._v = viewer
        self.title('Axis Limits')
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._build()
        self.update_states()

    # ── Construction ────────────────────────────────────────────────────────

    def _build(self) -> None:
        v = self._v
        f = ttk.Frame(self, padding=(12, 10))
        f.grid(row=0, column=0, sticky='nsew')

        for col, text in enumerate(('', 'Lo', 'Hi', 'Auto')):
            ttk.Label(f, text=text, anchor='center',
                      font=('TkDefaultFont', 9, 'bold')).grid(
                row=0, column=col, padx=4, pady=(0, 4))

        def _row(r, label, lo_var, hi_var, auto_var, which,
                 sb_from, sb_to, sb_inc, sb_fmt):
            ttk.Label(f, text=label, anchor='w').grid(
                row=r, column=0, sticky='w', padx=(0, 8), pady=3)
            lo_sb = ttk.Spinbox(f, textvariable=lo_var, width=9,
                                from_=sb_from, to=sb_to,
                                increment=sb_inc, format=sb_fmt)
            lo_sb.grid(row=r, column=1, padx=3)
            hi_sb = ttk.Spinbox(f, textvariable=hi_var, width=9,
                                from_=sb_from, to=sb_to,
                                increment=sb_inc, format=sb_fmt)
            hi_sb.grid(row=r, column=2, padx=3)
            auto_cb = ttk.Checkbutton(f, variable=auto_var,
                                      command=lambda w=which: self._on_auto(w))
            auto_cb.grid(row=r, column=3, padx=(8, 0))
            for sb in (lo_sb, hi_sb):
                sb.configure(command=v._an_apply_and_redraw)
                sb.bind('<Return>',   lambda _e: v._an_apply_and_redraw())
                sb.bind('<FocusOut>', lambda _e: v._an_apply_and_redraw())
            return (lo_sb, hi_sb), auto_cb

        self._x_sbs,     self._x_auto_cb     = _row(
            1, 'X (all panels):',
            v._an_xlim_lo_var, v._an_xlim_hi_var, v._an_xlim_auto,
            'x', 0, 10000, 10, '%.0f')

        ttk.Separator(f, orient='horizontal').grid(
            row=2, column=0, columnspan=4, sticky='ew', pady=6)

        self._ytop_sbs,   self._ytop_auto_cb   = _row(
            3, 'Y — overlay:',
            v._an_ylim_top_lo_var, v._an_ylim_top_hi_var, v._an_ylim_top_auto,
            'y_top', -2.0, 5.0, 0.05, '%.2f')

        self._ymain_sbs,  self._ymain_auto_cb  = _row(
            4, 'Y — main:',
            v._an_ylim_main_lo_var, v._an_ylim_main_hi_var, v._an_ylim_main_auto,
            'y_main', -2.0, 5.0, 0.05, '%.2f')

        self._yresid_sbs, self._yresid_auto_cb = _row(
            5, 'Y — residual:',
            v._an_ylim_resid_lo_var, v._an_ylim_resid_hi_var, v._an_ylim_resid_auto,
            'y_resid', -2.0, 2.0, 0.01, '%.3f')

        ttk.Separator(f, orient='horizontal').grid(
            row=6, column=0, columnspan=4, sticky='ew', pady=6)

        btn_row = ttk.Frame(f)
        btn_row.grid(row=7, column=0, columnspan=4, sticky='ew')
        ttk.Button(btn_row, text='Reset all', command=self._reset_all).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(btn_row, text='Close', command=self._on_close).pack(
            side=tk.LEFT, expand=True, fill=tk.X)

    # ── Callbacks ───────────────────────────────────────────────────────────

    def _on_auto(self, which: str) -> None:
        self._v._an_on_auto_toggle(which)
        self.update_states()

    def _reset_all(self) -> None:
        v = self._v
        for var in (v._an_xlim_auto, v._an_ylim_top_auto,
                    v._an_ylim_main_auto, v._an_ylim_resid_auto):
            var.set(True)
        for var in (v._an_xlim_lo_var, v._an_xlim_hi_var,
                    v._an_ylim_top_lo_var, v._an_ylim_top_hi_var,
                    v._an_ylim_main_lo_var, v._an_ylim_main_hi_var,
                    v._an_ylim_resid_lo_var, v._an_ylim_resid_hi_var):
            var.set('')
        self.update_states()
        v._an_refresh_plot()

    def _on_close(self) -> None:
        self._v._an_limits_dialog = None
        self.destroy()

    # ── State management ────────────────────────────────────────────────────

    def update_states(self) -> None:
        """Sync spinbox/checkbox enable state with current auto flags and axis availability."""
        v = self._v

        def _set(sbs, auto_var, has_axis):
            auto  = auto_var.get()
            sb_st = 'disabled' if (auto or not has_axis) else 'normal'
            cb_st = 'normal'   if has_axis                else 'disabled'
            for sb in sbs:
                sb.config(state=sb_st)

        _set(self._x_sbs,     v._an_xlim_auto,      v._an_ax_main is not None)
        _set(self._ytop_sbs,  v._an_ylim_top_auto,  v._an_ax_top  is not None)
        _set(self._ymain_sbs, v._an_ylim_main_auto, v._an_ax_main is not None)
        _set(self._yresid_sbs,v._an_ylim_resid_auto,v._an_ax_resid is not None)


class InstrumentMetricsDialog(tk.Toplevel):
    """
    Persistent (non-modal) time-series plot of instrument temperature channels.

    Channels 103–107 (ch3–ch7) are plotted vs. measurement time.  Each
    sample's acquisition time is marked with a vertical dashed line and
    labelled.  Individual channels can be toggled via Checkbuttons.
    """

    _CH_COLS  = ('channel_103', 'channel_104', 'channel_105', 'channel_106', 'channel_107')
    _CH_NAMES = tuple(_CHANNEL_LABELS[n] for n in (103, 104, 105, 106, 107))

    def __init__(self, viewer: 'EmissionLWIR',
                 labels: list, notes: dict) -> None:
        super().__init__(viewer)
        self._v      = viewer
        self._labels = labels
        self._notes  = notes
        self.title('Instrument Metrics')
        self.resizable(True, True)
        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._ch_vars: dict[str, tk.BooleanVar] = {}
        self._build()
        self._draw()

    def _build(self) -> None:
        # Temperature channel toggles (primary axis)
        ctrl = ttk.Frame(self)
        ctrl.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(ctrl, text='Temperature:').pack(side=tk.LEFT, padx=(0, 6))
        for col, name in zip(self._CH_COLS, self._CH_NAMES):
            var = tk.BooleanVar(value=True)
            self._ch_vars[col] = var
            ttk.Checkbutton(ctrl, text=name, variable=var,
                            command=self._draw).pack(side=tk.LEFT, padx=2)

        # BB resistance toggles (secondary axis)
        ctrl2 = ttk.Frame(self)
        ctrl2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(ctrl2, text='BB Resistance:').pack(side=tk.LEFT, padx=(0, 6))
        for key, num in (('bb_ch101', 101), ('bb_ch102', 102)):
            var = tk.BooleanVar(value=True)
            self._ch_vars[key] = var
            ttk.Checkbutton(ctrl2, text=_CHANNEL_LABELS[num], variable=var,
                            command=self._draw).pack(side=tk.LEFT, padx=2)

        # Matplotlib canvas — primary + twin secondary axis created once
        fig_frame = ttk.Frame(self)
        fig_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self._fig, self._ax = plt.subplots(figsize=(10, 4))
        self._ax2 = self._ax.twinx()
        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, fig_frame)

        ttk.Button(self, text='Close', command=self._on_close).pack(
            side=tk.BOTTOM, pady=(0, 6))

    def _draw(self) -> None:
        import pandas as pd
        import matplotlib.dates as mdates
        ax  = self._ax
        ax2 = self._ax2
        ax.cla()
        ax2.cla()
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()

        def _parse_sort(dtime_raw) -> tuple[list | None, np.ndarray | None]:
            """Return (sorted valid-Timestamps list, index array into original) or (None, None).

            NaT entries are silently excluded so matplotlib never receives a
            mixed Timestamp/NaT array.
            """
            try:
                parsed = pd.to_datetime(list(dtime_raw), errors='coerce', format='mixed')
                valid  = np.array([pd.notna(t) for t in parsed])
                if valid.any():
                    valid_idx = np.where(valid)[0]
                    sort_within = np.argsort(
                        [parsed[i].value for i in valid_idx], kind='stable'
                    )
                    final_idx = valid_idx[sort_within]
                    return [parsed[i] for i in final_idx], final_idx
            except Exception:
                pass
            return None, None

        # ── Primary axis: temperature channels 103–107 ──────────────────────
        dtimes_list, sort_idx = _parse_sort(self._notes.get('dtime', []))

        sorted_labels = ([self._labels[i] for i in sort_idx]
                         if sort_idx is not None else list(self._labels))
        x_values = dtimes_list if dtimes_list is not None else list(range(len(self._labels)))

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        any_temp = False
        for c_idx, (col, name) in enumerate(zip(self._CH_COLS, self._CH_NAMES)):
            if not self._ch_vars[col].get():
                continue
            vals_raw = np.array(list(self._notes.get(col, [])), dtype=float)
            if not np.isfinite(vals_raw).any():
                continue
            vals = vals_raw[sort_idx] if sort_idx is not None else vals_raw
            ax.plot(x_values, vals, label=name,
                    color=colors[c_idx % len(colors)], marker='o', ms=4, lw=1.2)
            any_temp = True

        # Per-sample annotations: shaded window (±1 min), vertical line, label
        if dtimes_list is not None and any_temp:
            xform    = ax.get_xaxis_transform()
            half_dur = pd.Timedelta(minutes=1)
            for lbl, dt in zip(sorted_labels, dtimes_list):
                if pd.isna(dt):
                    continue
                ax.axvspan(dt - half_dur, dt + half_dur,
                           color='gray', alpha=0.10, lw=0, zorder=0)
                ax.axvline(dt, color='gray', lw=0.7, ls='--', zorder=1)
                ax.text(dt, 1.0, lbl, rotation=90, va='top', ha='right',
                        fontsize=7, color='gray', transform=xform)

        # ── Secondary axis: BB resistance channels 101–102 (Ω) ───────────────
        bb_dtimes_list, bb_sort_idx = _parse_sort(self._notes.get('bb_dtime', []))
        bb_x = bb_dtimes_list if bb_dtimes_list is not None else list(range(len(
            self._notes.get('bb_dtime', []))))

        any_resist = False
        bb_color_offset = len(self._CH_COLS)
        for b_idx, (key, num) in enumerate((('bb_ch101', 101), ('bb_ch102', 102))):
            if not self._ch_vars[key].get():
                continue
            raw = np.array(list(self._notes.get(key, [])), dtype=float)
            if not np.isfinite(raw).any():
                continue
            vals = raw[bb_sort_idx] if bb_sort_idx is not None else raw
            color = colors[(bb_color_offset + b_idx) % len(colors)]
            ax2.plot(bb_x, vals, label=_CHANNEL_LABELS[num],
                     color=color, marker='s', ms=5, lw=1.2, ls='--')
            any_resist = True

        # ── Axis labels, legend, formatting ──────────────────────────────────
        use_dates = dtimes_list is not None or bb_dtimes_list is not None
        if use_dates:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        if any_resist:
            ax2.set_ylabel('Resistance (Ω)')
        ax2.tick_params(axis='y', which='both',
                        right=any_resist, labelright=any_resist)

        # Combine legends from both axes
        lines1, lbls1 = ax.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, loc='upper right')

        if use_dates:
            self._fig.autofmt_xdate(rotation=30)
        ax.grid(True, lw=0.4, alpha=0.5)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def _on_close(self) -> None:
        self._v._metrics_dialog = None
        plt.close(self._fig)
        self.destroy()


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class EmissionLWIR(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title(f'EmissionLWIR  v{__version__}')
        _set_window_size(self, fraction=0.85, min_w=1100, min_h=650)
        self._sw = self.winfo_screenwidth()

        # ── Processing state ────────────────────────────────────────────────
        self._fdir:         str | None  = None
        self._fdirs:        list[str]   = []     # non-empty → multi-folder mode
        self._last_load_dir: str | None = None   # dir of last loaded results/data → save default
        self._raw_data:     dict | None = None   # SBM loaded on folder select
        self._emcal_result: dict | None = None
        self._sma_result:   dict | None = None

        # Per-folder intermediate results (multi-folder mode only)
        self._individual_emcal_results:  list[dict] = []
        self._individual_calrad_results: list[dict] = []
        self._folder_sbm_bbs:  list[dict] = []   # per-folder {'bbh': ..., 'bbc': ...}
        self._label_folder_map: dict[str, int] = {}  # merged label → folder index

        self._calrad_opts: dict = dict(_CALRAD_DEFAULTS)
        self._emcal_opts:  dict = dict(_EMCAL_DEFAULTS)
        self._sma_opts:    dict = dict(_SMA_DEFAULTS)

        # ── Data tab state ──────────────────────────────────────────────────
        self._data_mode:  str = 'sbm'    # 'sbm' | 'radiance' | 'emissivity'
        self._data_label: str | None = None
        self._data_page:  int = 0
        self._data_secax = None
        self._notes_df   = None          # cached notes DataFrame; reset on folder load
        self._metrics_dialog: object = None

        # ── Speclib figure secondary axis ────────────────────────────────────
        self._sl_secax = None

        # ── Speclib tab state ────────────────────────────────────────────────
        self._full_library:  dict = {}
        self._current_album: dict = {}
        self._extra_libs:    dict = {}
        self._browse_source: dict = {}
        self._sl_lib_path:   str  = ''

        self._active_sid:    int | None = None
        self._excluded_sids: set = set()
        self._forced_sids:   set = set()
        self._filtered_ids:  list[int] = []
        self._sl_page:       int = 0

        # ── Analysis tab state ──────────────────────────────────────────────
        self._sma_label:           str | None = None
        self._sma_lib_unit:        dict       = {}
        self._sma_group_lib_unit:  dict       = {}
        self._sma_lib_raw:         dict       = {}
        self._sma_group_lib_raw:   dict       = {}
        self._an_group_var     = tk.BooleanVar(value=False)
        self._an_residual_var  = tk.BooleanVar(value=False)
        self._an_error_var     = tk.BooleanVar(value=False)
        self._an_cumulative_var = tk.BooleanVar(value=True)
        self._an_other_var     = tk.BooleanVar(value=False)
        self._an_threshold_var = tk.DoubleVar(value=5.0)
        self._an_offset_var    = tk.DoubleVar(value=0.0)

        # ── Axis limit controls ──────────────────────────────────────────────
        self._an_xlim_auto           = tk.BooleanVar(value=True)
        self._an_xlim_lo_var         = tk.StringVar(value='')
        self._an_xlim_hi_var         = tk.StringVar(value='')
        self._an_ylim_top_auto       = tk.BooleanVar(value=True)
        self._an_ylim_top_lo_var     = tk.StringVar(value='')
        self._an_ylim_top_hi_var     = tk.StringVar(value='')
        self._an_ylim_main_auto      = tk.BooleanVar(value=True)
        self._an_ylim_main_lo_var    = tk.StringVar(value='')
        self._an_ylim_main_hi_var    = tk.StringVar(value='')
        self._an_ylim_resid_auto     = tk.BooleanVar(value=True)
        self._an_ylim_resid_lo_var   = tk.StringVar(value='')
        self._an_ylim_resid_hi_var   = tk.StringVar(value='')
        self._an_ax_top:    object   = None   # set each redraw; used for limit read-back
        self._an_ax_main:   object   = None
        self._an_ax_resid:  object   = None
        self._an_limits_dialog:object = None

        # ── Color scheme ────────────────────────────────────────────────────
        self._color_scheme_var = tk.StringVar(value=_COLOR_SCHEME_DEFAULT)
        self._an_colors: list  = _COLOR_SCHEMES[_COLOR_SCHEME_DEFAULT]

        self._build_ui()
        self._refresh_toolbar()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _on_close(self) -> None:
        plt.close('all')
        self.destroy()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))
        self._notebook = notebook

        self._tab_data     = ttk.Frame(notebook)
        self._tab_speclib  = ttk.Frame(notebook)
        self._tab_analysis = ttk.Frame(notebook)
        notebook.add(self._tab_data,     text='Data')
        notebook.add(self._tab_speclib,  text='Speclib')
        notebook.add(self._tab_analysis, text='Analysis')

        self._build_data_tab()
        self._build_speclib_tab()
        self._build_analysis_tab()

        # Window-level arrow-key navigation (active tab determines target)
        self.bind('<Up>',   lambda _e: self._on_nav_key(-1))
        self.bind('<Down>', lambda _e: self._on_nav_key(1))

    # ── Toolbar ─────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        # ── Row 1: data / library / save ────────────────────────────────────
        row1 = ttk.Frame(outer)
        row1.pack(side=tk.TOP, fill=tk.X)

        self._btn_load_data = ttk.Button(
            row1, text='Load Data', command=self._on_load_data)
        self._btn_load_data.pack(side=tk.LEFT, padx=2)

        self._btn_load_results = ttk.Button(
            row1, text='Load Results', command=self._on_load_results)
        self._btn_load_results.pack(side=tk.LEFT, padx=2)

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        self._btn_load_lib = ttk.Button(
            row1, text='Load Library', command=self._on_load_library)
        self._btn_load_lib.pack(side=tk.LEFT, padx=2)

        self._btn_build_lib = ttk.Button(
            row1, text='Build Library', command=self._on_build_library)
        self._btn_build_lib.pack(side=tk.LEFT, padx=2)

        self._btn_save_results = ttk.Button(
            row1, text='Save Results', command=self._on_save_results)
        self._btn_save_results.pack(side=tk.RIGHT, padx=2)

        # Optional filename suffix; blank → timestamp (see _on_save_results).
        self._suffix_var = tk.StringVar(value='')
        self._suffix_entry = ttk.Entry(row1, textvariable=self._suffix_var, width=18)
        self._suffix_entry.pack(side=tk.RIGHT, padx=2)
        ttk.Label(row1, text='Suffix (leave blank for a timestamp):').pack(side=tk.RIGHT, padx=(6, 2))

        ttk.Separator(row1, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=6, pady=2)

        # ── Row 2: processing + status ───────────────────────────────────────
        row2 = ttk.Frame(outer)
        row2.pack(side=tk.TOP, fill=tk.X, pady=(2, 0))

        self._btn_calrad = ttk.Button(
            row2, text='cal_rad()', command=self._on_run_cal_rad)
        self._btn_calrad.pack(side=tk.LEFT, padx=2)

        self._btn_emcal = ttk.Button(
            row2, text='emcal()', command=self._on_run_emcal)
        self._btn_emcal.pack(side=tk.LEFT, padx=2)

        self._btn_sma = ttk.Button(
            row2, text='sma()', command=self._on_run_sma)
        self._btn_sma.pack(side=tk.LEFT, padx=2)

        self._status_var = tk.StringVar(value='No data loaded')
        ttk.Label(row2, textvariable=self._status_var,
                  foreground='gray').pack(side=tk.LEFT, padx=14)

        # Color scheme selector (right side of row2)
        self._cb_color_scheme = ttk.Combobox(
            row2,
            textvariable=self._color_scheme_var,
            values=list(_COLOR_SCHEMES.keys()),
            state='readonly',
            width=10,
        )
        self._cb_color_scheme.pack(side=tk.RIGHT, padx=(0, 6))
        self._cb_color_scheme.bind('<<ComboboxSelected>>', self._on_color_scheme_change)
        ttk.Label(row2, text='Color Scheme:').pack(side=tk.RIGHT, padx=(0, 2))

    def _refresh_toolbar(self) -> None:
        have_data   = self._fdir is not None or bool(self._fdirs)
        have_calrad = (self._emcal_result is not None
                       and bool(self._emcal_result.get('rad', {})))
        have_emiss  = (self._emcal_result is not None
                       and bool(self._emcal_result.get('emiss', {})))
        have_emcal  = self._emcal_result is not None
        have_lib    = bool(self._full_library) or bool(self._extra_libs)
        have_sma    = self._sma_result is not None

        def _state(flag): return 'normal' if flag else 'disabled'

        self._btn_calrad['state']       = _state(have_data)
        self._btn_emcal['state']        = _state(have_data)
        self._btn_sma['state']          = _state(have_emiss and have_lib)
        self._btn_save_results['state'] = _state(have_emcal or have_sma)

        self._rb_sbm['state'] = _state(self._raw_data is not None)
        self._rb_rad['state'] = _state(have_calrad)
        self._rb_em['state']  = _state(have_emiss)
        self._btn_data_delete['state'] = _state(have_emcal or self._raw_data is not None)
        self._btn_an_delete['state']   = _state(have_sma)
        in_radiance = have_calrad and self._data_mode_var.get() == 'radiance'
        self._cb_show_model['state']  = _state(in_radiance)
        self._rb_rad_raw['state']     = _state(in_radiance)
        self._rb_rad_corr['state']    = _state(in_radiance)

        parts = []
        if self._fdirs:
            parts.append(f'{Path(self._fdirs[0]).parent.name}  ({len(self._fdirs)} folders)')
        elif self._fdir:
            parts.append(Path(self._fdir).name)
        if have_calrad and not have_emiss: parts.append('cal_rad ✓')
        if have_emiss:    parts.append('emcal ✓')
        if have_lib:      parts.append('library ✓')
        if have_sma:      parts.append('sma ✓')
        self._status_var.set('  |  '.join(parts) if parts else 'No data loaded')

    def _on_color_scheme_change(self, _event=None) -> None:
        name = self._color_scheme_var.get()
        self._an_colors = _COLOR_SCHEMES[name]
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self._an_colors)
        if self._emcal_result is not None or self._raw_data is not None:
            self._refresh_data_plot()
        if self._sma_result is not None:
            self._an_refresh_plot()

    # ── Data tab ────────────────────────────────────────────────────────────

    def _build_data_tab(self) -> None:
        paned = ttk.PanedWindow(self._tab_data, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: sample list + radiance/emissivity toggle
        left = ttk.Frame(paned, width=int(self._sw * 0.13))
        left.pack_propagate(False)
        paned.add(left, weight=0)

        lf = ttk.LabelFrame(left, text='Samples', padding=4)
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self._data_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set, selectmode=tk.SINGLE, exportselection=False,
            font=('TkFixedFont', 13))
        sb.config(command=self._data_listbox.yview)
        self._data_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._data_listbox.bind('<<ListboxSelect>>', self._on_data_select)

        ctrl = ttk.Frame(left)
        ctrl.pack(fill=tk.X, padx=4, pady=4)
        self._data_mode_var = tk.StringVar(value='sbm')
        self._rb_sbm = ttk.Radiobutton(ctrl, text='Single-beam', variable=self._data_mode_var,
                        value='sbm',        command=self._on_data_mode)
        self._rb_sbm.pack(anchor=tk.W)
        self._rb_rad = ttk.Radiobutton(ctrl, text='Radiance',    variable=self._data_mode_var,
                        value='radiance',   command=self._on_data_mode)
        self._rb_rad.pack(anchor=tk.W)
        self._rb_em  = ttk.Radiobutton(ctrl, text='Emissivity',  variable=self._data_mode_var,
                        value='emissivity', command=self._on_data_mode)
        self._rb_em.pack(anchor=tk.W)
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self._show_model_var = tk.BooleanVar(value=False)
        self._cb_show_model  = ttk.Checkbutton(ctrl, text='Show model fit',
                                               variable=self._show_model_var,
                                               command=self._refresh_data_plot)
        self._cb_show_model.pack(anchor=tk.W)
        self._show_bb_var = tk.BooleanVar(value=True)
        self._cb_show_bb  = ttk.Checkbutton(ctrl, text='Show BBs',
                                            variable=self._show_bb_var,
                                            command=self._refresh_data_plot)
        self._cb_show_bb.pack(anchor=tk.W)
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, text='Radiance display:').pack(anchor=tk.W)
        self._rad_display_var = tk.StringVar(value='raw')
        self._rb_rad_raw  = ttk.Radiobutton(ctrl, text='Raw',
                                             variable=self._rad_display_var, value='raw',
                                             command=self._refresh_data_plot)
        self._rb_rad_raw.pack(anchor=tk.W)
        self._rb_rad_corr = ttk.Radiobutton(ctrl, text='Corrected (÷ max_ε, −dw)',
                                             variable=self._rad_display_var, value='corrected',
                                             command=self._refresh_data_plot)
        self._rb_rad_corr.pack(anchor=tk.W)
        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self._btn_data_delete = ttk.Button(
            ctrl, text='Delete spectrum', state=tk.DISABLED,
            command=lambda: self._on_delete_sample(self._data_label))
        self._btn_data_delete.pack(fill=tk.X)

        # Right: nav bar + plot + info panel
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._data_plot_var = tk.StringVar(value='stacked')
        self._data_nav_label = self._build_nav_bar(
            right,
            plot_var   = self._data_plot_var,
            plot_cmd   = self._on_data_plot_mode,
            prev_cmd   = self._on_data_prev,
            next_cmd   = self._on_data_next,
        )

        hpaned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        hpaned.pack(fill=tk.BOTH, expand=True)

        plot_frame = ttk.Frame(hpaned)
        hpaned.add(plot_frame, weight=3)
        self._data_fig    = Figure(figsize=(9, 5))
        self._data_ax     = self._data_fig.add_subplot(111)
        self._data_fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)
        self._data_canvas = FigureCanvasTkAgg(self._data_fig, master=plot_frame)
        self._data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._data_canvas, plot_frame)

        info_frame = ttk.LabelFrame(hpaned, text='Measurement Info')
        hpaned.add(info_frame, weight=1)
        self._info_tree = ttk.Treeview(
            info_frame, columns=('field', 'value'), show='tree headings', height=5)
        self._info_tree.heading('#0',    text='')
        self._info_tree.heading('field', text='Field')
        self._info_tree.heading('value', text='Value')
        self._info_tree.column('#0',    width=14, minwidth=14, stretch=False)
        self._info_tree.column('field', width=150, stretch=False)
        self._info_tree.column('value', width=150, stretch=True)
        self._info_tree.tag_configure('header', font=('TkDefaultFont', 9, 'bold'))
        isb = ttk.Scrollbar(info_frame, orient=tk.VERTICAL,
                             command=self._info_tree.yview)
        self._info_tree.configure(yscrollcommand=isb.set)
        # Button packed first (BOTTOM) so tree's expand=True doesn't crowd it
        self._btn_instrument_metrics = ttk.Button(
            info_frame, text='Plot Instrument Metrics',
            command=self._on_plot_instrument_metrics, state=tk.DISABLED)
        self._btn_instrument_metrics.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=(2, 4))
        self._info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        isb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Speclib tab ─────────────────────────────────────────────────────────

    def _build_speclib_tab(self) -> None:
        # Top: source selector
        top = ttk.Frame(self._tab_speclib)
        top.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        self._sl_status = ttk.Label(top, text='No library loaded', foreground='gray')
        self._sl_status.pack(side=tk.LEFT, padx=4)

        # Paned: left panel + plot
        paned = ttk.PanedWindow(self._tab_speclib, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        left = ttk.Frame(paned, width=int(self._sw * 0.13))
        left.pack_propagate(False)
        paned.add(left, weight=0)

        # Spectrum listbox
        lf = ttk.LabelFrame(left, text='Spectra', padding=4)
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self._sl_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set, selectmode=tk.SINGLE,
            exportselection=False, font=('TkFixedFont', 13))
        sb.config(command=self._sl_listbox.yview)
        self._sl_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._sl_listbox.bind('<<ListboxSelect>>', self._sl_on_select)

        # Action buttons (stacked vertically)
        abf = ttk.Frame(left)
        abf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(abf, text='Toggle Excluded',
                   command=self._sl_toggle_excluded).pack(fill=tk.X, pady=1)
        ttk.Button(abf, text='Toggle Forced',
                   command=self._sl_toggle_forced).pack(fill=tk.X, pady=1)
        ttk.Button(abf, text='Remove',
                   command=self._sl_remove_spectrum).pack(fill=tk.X, pady=1)

        # Endmember count display
        self._sl_em_status = ttk.Label(left, text='', foreground='#555')
        self._sl_em_status.pack(anchor=tk.W, padx=6, pady=(0, 4))

        # Right: nav bar + plot + info panel
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._sl_plot_var = tk.StringVar(value='individual')
        self._sl_nav_label = self._build_nav_bar(
            right,
            plot_var = self._sl_plot_var,
            plot_cmd = self._sl_refresh_plot,
            prev_cmd = self._on_sl_prev,
            next_cmd = self._on_sl_next,
        )

        sl_hpaned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        sl_hpaned.pack(fill=tk.BOTH, expand=True)

        sl_plot_frame = ttk.Frame(sl_hpaned)
        sl_hpaned.add(sl_plot_frame, weight=3)
        self._sl_fig    = Figure(figsize=(9, 6))
        self._sl_ax     = self._sl_fig.add_subplot(111)
        self._sl_fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)
        self._sl_canvas = FigureCanvasTkAgg(self._sl_fig, master=sl_plot_frame)
        self._sl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._sl_canvas, sl_plot_frame)

        sl_info_frame = ttk.LabelFrame(sl_hpaned, text='Endmember Info')
        sl_hpaned.add(sl_info_frame, weight=1)
        self._sl_info_tree = ttk.Treeview(
            sl_info_frame, columns=('field', 'value'),
            show='tree headings', height=5)
        self._sl_info_tree.heading('#0',    text='')
        self._sl_info_tree.heading('field', text='Field')
        self._sl_info_tree.heading('value', text='Value')
        self._sl_info_tree.column('#0',    width=14, minwidth=14, stretch=False)
        self._sl_info_tree.column('field', width=160, stretch=False)
        self._sl_info_tree.column('value', width=160, stretch=True)
        self._sl_info_tree.tag_configure('header', font=('TkDefaultFont', 9, 'bold'))
        sl_isb = ttk.Scrollbar(sl_info_frame, orient=tk.VERTICAL,
                               command=self._sl_info_tree.yview)
        self._sl_info_tree.configure(yscrollcommand=sl_isb.set)
        self._sl_info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sl_isb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Analysis tab ────────────────────────────────────────────────────────

    def _build_analysis_tab(self) -> None:
        paned = ttk.PanedWindow(self._tab_analysis, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: sample list + action buttons
        left = ttk.Frame(paned, width=int(self._sw * 0.13))
        left.pack_propagate(False)
        paned.add(left, weight=0)

        lf = ttk.LabelFrame(left, text='Samples', padding=4)
        lf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)
        sb = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        self._sma_listbox = tk.Listbox(
            lf, yscrollcommand=sb.set, selectmode=tk.SINGLE, exportselection=False,
            font=('TkFixedFont', 13))
        sb.config(command=self._sma_listbox.yview)
        self._sma_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._sma_listbox.bind('<<ListboxSelect>>', self._an_on_select)

        bf = ttk.Frame(left)
        bf.pack(fill=tk.X, padx=4, pady=4)
        ttk.Button(bf, text='Save current plot', command=self._an_save_current).pack(fill=tk.X, pady=1)
        ttk.Button(bf, text='Save all plots',     command=self._an_save_all).pack(fill=tk.X, pady=1)
        self._btn_an_delete = ttk.Button(
            bf, text='Delete spectrum', state=tk.DISABLED,
            command=lambda: self._on_delete_sample(self._sma_label))
        self._btn_an_delete.pack(fill=tk.X, pady=1)

        # Right: nav/opts bar spanning full width, then figure | pie+table
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # Nav bar spans full width; display options centered between the arrows
        nav_bar = ttk.Frame(right)
        nav_bar.pack(fill=tk.X, padx=4, pady=(4, 2))
        ttk.Button(nav_bar, text='◀', width=3,
                   command=self._on_an_prev).pack(side=tk.LEFT)
        ttk.Button(nav_bar, text='▶', width=3,
                   command=self._on_an_next).pack(side=tk.RIGHT)
        ttk.Separator(nav_bar, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=6, pady=2)
        ttk.Button(nav_bar, text='Limits…',
                   command=self._an_open_limits_dialog).pack(side=tk.RIGHT, padx=4)
        opts_center = ttk.Frame(nav_bar)
        opts_center.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Checkbutton(opts_center, text='Group',
                        variable=self._an_group_var,
                        command=self._an_refresh_plot).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(opts_center, text='Residual',
                        variable=self._an_residual_var,
                        command=self._an_refresh_plot).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(opts_center, text='Errors',
                        variable=self._an_error_var,
                        command=self._an_refresh_plot).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(opts_center, text='Cumulative',
                        variable=self._an_cumulative_var,
                        command=self._an_refresh_plot).pack(side=tk.LEFT, expand=True)
        ttk.Checkbutton(opts_center, text='Other',
                        variable=self._an_other_var,
                        command=self._an_refresh_plot).pack(side=tk.LEFT, expand=True)
        ttk.Label(opts_center, text='Threshold (%)').pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(
            opts_center,
            textvariable=self._an_threshold_var,
            from_=0.0, to=50.0, increment=1.0,
            width=4, format='%.0f',
            command=self._an_refresh_plot,
        ).pack(side=tk.LEFT)
        self._an_threshold_var.trace_add(
            'write', lambda *_: self._an_refresh_plot())
        ttk.Label(opts_center, text='Offset').pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(
            opts_center,
            textvariable=self._an_offset_var,
            from_=0.0, to=1.0, increment=0.02,
            width=5, format='%.2f',
            command=self._an_refresh_plot,
        ).pack(side=tk.LEFT)
        self._an_offset_var.trace_add(
            'write', lambda *_: self._an_refresh_plot())

        h_paned = ttk.PanedWindow(right, orient=tk.HORIZONTAL)
        h_paned.pack(fill=tk.BOTH, expand=True)

        # Left pane — spectrum figure, full height
        spec_frame = ttk.Frame(h_paned)
        h_paned.add(spec_frame, weight=3)
        self._an_fig    = Figure(figsize=(7, 5))
        self._an_canvas = FigureCanvasTkAgg(self._an_fig, master=spec_frame)
        self._an_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._an_canvas, spec_frame)

        # Right pane — pie chart (top ~1/3) + concentrations table (bottom ~2/3)
        right_panel = ttk.Frame(h_paned)
        h_paned.add(right_panel, weight=1)

        self._an_right_vpaned = ttk.PanedWindow(right_panel, orient=tk.VERTICAL)
        self._an_right_vpaned.pack(fill=tk.BOTH, expand=True)

        pie_frame = ttk.Frame(self._an_right_vpaned)
        self._an_right_vpaned.add(pie_frame, weight=1)
        self._an_pie_fig    = Figure(figsize=(3, 3))
        self._an_pie_canvas = FigureCanvasTkAgg(self._an_pie_fig, master=pie_frame)
        self._an_pie_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        aux_frame = ttk.LabelFrame(self._an_right_vpaned, text='Concentrations & Metrics')
        self._an_right_vpaned.add(aux_frame, weight=2)
        self._an_tree = ttk.Treeview(aux_frame, show='headings', height=5)
        aux_sb = ttk.Scrollbar(aux_frame, orient=tk.VERTICAL,
                               command=self._an_tree.yview)
        self._an_tree.configure(yscrollcommand=aux_sb.set)
        self._an_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        aux_sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Set initial sash so pie occupies ~1/3 of right panel height
        self.after(150, self._an_init_sash)

    # ── Shared nav bar ──────────────────────────────────────────────────────

    def _build_nav_bar(
        self,
        parent: tk.Widget,
        plot_var: tk.StringVar,
        plot_cmd,
        prev_cmd,
        next_cmd,
    ) -> tk.StringVar:
        """Build [◀] [Individual] [Stacked] <center label> [▶] above a plot area."""
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, padx=4, pady=(4, 0))

        ttk.Button(bar, text='◀', width=3, command=prev_cmd).pack(side=tk.LEFT)
        ttk.Radiobutton(bar, text='Individual', variable=plot_var,
                        value='individual', command=plot_cmd).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(bar, text='Stacked', variable=plot_var,
                        value='stacked', command=plot_cmd).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text='▶', width=3, command=next_cmd).pack(side=tk.RIGHT)

        label_var = tk.StringVar(value='')
        ttk.Label(bar, textvariable=label_var, anchor='center',
                  foreground='#444').pack(side=tk.LEFT, fill=tk.X, expand=True)
        return label_var

    # -----------------------------------------------------------------------
    # Toolbar actions
    # -----------------------------------------------------------------------

    def _on_load_data(self) -> None:
        path = filedialog.askdirectory(title='Select measurement folder')
        if not path:
            return
        self._last_load_dir = path

        _EXCLUDE = {'sma_plots'}
        subdirs = sorted(
            f.path for f in os.scandir(path)
            if f.is_dir() and not f.name.startswith('.') and f.name not in _EXCLUDE
        )

        self._raw_data     = None
        self._emcal_result = None
        self._notes_df     = None
        self._individual_emcal_results  = []
        self._individual_calrad_results = []
        self._folder_sbm_bbs   = []
        self._label_folder_map = {}
        self._data_label   = None
        self._data_page    = 0
        self._clear_info_tree()
        self._clear_analysis()

        if subdirs:
            dlg = FolderSelectDialog(self, path, preselected=subdirs)
            if dlg.result is None:
                return
            self._fdirs = dlg.result
            self._fdir  = None
            self._data_ax.cla()
            self._data_ax.text(
                0.5, 0.5,
                f'Loading SBM from {len(self._fdirs)} folder(s) ...',
                ha='center', va='center', transform=self._data_ax.transAxes,
                color='gray', fontsize=11,
            )
            self._data_canvas.draw_idle()
            self._refresh_toolbar()
            self._notebook.select(self._tab_data)
            self._load_folders_raw()
        else:
            self._fdir  = path
            self._fdirs = []
            self._refresh_toolbar()
            self._notebook.select(self._tab_data)
            self._load_folder_raw(path)

    def _on_load_folders(self) -> None:
        parent = filedialog.askdirectory(title='Select parent folder containing measurement subfolders')
        if not parent:
            return
        subdirs = sorted(f.path for f in os.scandir(parent) if f.is_dir())
        if not subdirs:
            messagebox.showwarning('No subfolders', f'No subfolders found in:\n{parent}')
            return
        dlg = FolderSelectDialog(self, parent, preselected=subdirs)
        if dlg.result is None:
            return
        self._fdirs        = dlg.result
        self._fdir         = None
        self._raw_data     = None
        self._emcal_result = None
        self._notes_df     = None
        self._individual_emcal_results  = []
        self._individual_calrad_results = []
        self._folder_sbm_bbs   = []
        self._label_folder_map = {}
        self._data_label   = None
        self._data_page    = 0
        self._clear_info_tree()
        self._clear_analysis()
        self._notebook.select(self._tab_data)
        self._data_ax.cla()
        self._data_ax.text(
            0.5, 0.5,
            f'Loading SBM from {len(self._fdirs)} folder(s) ...',
            ha='center', va='center', transform=self._data_ax.transAxes,
            color='gray', fontsize=11,
        )
        self._data_canvas.draw_idle()
        self._refresh_toolbar()
        self._load_folders_raw()

    def _on_load_results(self) -> None:
        path = filedialog.askopenfilename(
            title='Load Results',
            filetypes=[('HDF files', '*.hdf *.h5'), ('All files', '*')],
        )
        if not path:
            return
        try:
            d = readHDF(path)
        except Exception as exc:
            messagebox.showerror('Load failed', str(exc))
            logging.exception("readHDF failed for %s", path)
            return
        self._last_load_dir = os.path.dirname(path)

        # Detect result type — filename first, content fallback, user prompt last
        basename = os.path.basename(path)
        if 'emcal_results' in basename:
            kind = 'emcal'
        elif 'sma_results' in basename:
            kind = 'sma'
        elif 'method' in d:
            kind = 'emcal'
        elif 'algorithm' in d:
            kind = 'sma'
        elif 'data' in d and 'xaxis' in d and 'label' in d:
            kind = 'emiss_array'
        else:
            ans = messagebox.askquestion(
                'Ambiguous file',
                'Cannot determine result type from filename or content.\n\n'
                'Is this an emcal result?\n(No = treat as sma result)',
                icon='question',
            )
            kind = 'emcal' if ans == 'yes' else 'sma'

        if kind == 'emcal':
            self._load_emcal_result(d)
        elif kind == 'sma':
            self._load_sma_result(d, source_path=path)
        else:
            self._load_emiss_array(d)

    def _load_emcal_result(self, d: dict) -> None:
        if 'label' in d and hasattr(d['label'], 'tolist'):
            d['label'] = [str(s) for s in d['label']]
        self._fdir  = None
        self._fdirs = []
        self._individual_emcal_results  = []
        self._individual_calrad_results = []
        self._folder_sbm_bbs   = []
        self._label_folder_map = {}
        self._emcal_result = d
        self._clear_analysis()

        # Reconstruct _raw_data for SBM tab if sbm group is present
        sbm = d.get('sbm', {})
        if sbm:
            self._raw_data = {
                'wn':     d['xaxis'],
                'labels': d.get('label', []),
                'sbm':    sbm,
            }
        else:
            self._raw_data = None

        mode = 'radiance' if d.get('rad') else 'emissivity'
        self._data_mode_var.set(mode)
        self._data_mode = mode
        self._populate_data_tab()
        self._refresh_toolbar()
        self._notebook.select(self._tab_data)
        logging.info("Loaded emcal result: %d samples", len(d.get('label', [])))

    def _load_emiss_array(self, d: dict) -> None:
        """Wrap a flat emissivity array (data/xaxis/label) into an emcal-compatible dict."""
        labels = [str(s) for s in np.asarray(d['label'])]
        data   = np.asarray(d['data'])    # (n_samples, n_wn)
        xaxis  = np.asarray(d['xaxis'])
        wrapped = {
            'xaxis': xaxis,
            'label': labels,
            'data':  data,
            'emiss': {lbl: data[i] for i, lbl in enumerate(labels)},
        }
        self._load_emcal_result(wrapped)
        logging.info("Loaded emissivity array: %d spectra", len(labels))

    def _load_sma_result(self, d: dict, source_path: str = '') -> None:
        for key in ('sample_labels', 'labels', 'groups'):
            if key in d and hasattr(d[key], 'tolist'):
                d[key] = [str(s) for s in d[key]]
        if 'wn_range' in d and hasattr(d['wn_range'], 'tolist'):
            d['wn_range'] = tuple(float(x) for x in d['wn_range'])
        if 'has_slope' in d:
            d['has_slope'] = bool(d['has_slope'])

        self._fdir  = None
        self._fdirs = []
        self._individual_emcal_results  = []
        self._individual_calrad_results = []
        self._raw_data = None
        self._clear_analysis()
        self._sma_result = d

        # Reconstruct a minimal emcal_result for data tab (emissivity mode only)
        labels   = d.get('sample_labels', [])
        measured = d.get('measured')
        xaxis    = d.get('xaxis')
        if measured is not None and xaxis is not None and labels:
            self._emcal_result = {
                'xaxis':  xaxis,
                'label':  labels,
                'data':   measured,   # sma() reads em_data['data'] as the emissivity array
                'emiss':  {lbl: measured[i] for i, lbl in enumerate(labels)},
            }
        else:
            self._emcal_result = None

        # Restore endmember library into the Speclib tab
        # Works for both our per-entry format and the flat DaVinci struct format
        endlib = d.get('endlib')
        if endlib and isinstance(endlib, dict) and len(endlib) > 0:
            try:
                album = _to_album(endlib)
                self._full_library  = album
                self._browse_source = album
                self._sl_lib_path   = source_path
                self._sl_populate_listbox()
                self._sl_update_status()
                logging.info("Restored endmember library: %d spectra", len(album))
            except Exception as exc:
                logging.warning("Could not restore endlib from sma result: %s", exc)

        self._data_mode_var.set('emissivity')
        self._data_mode = 'emissivity'
        self._populate_data_tab()
        self._populate_analysis_tab()
        self._refresh_toolbar()
        self._an_refresh_plot()
        self._an_refresh_table()
        logging.info(
            "Loaded sma result: %d samples, %d endmembers",
            len(labels), len(d.get('labels', [])),
        )

    # -----------------------------------------------------------------------
    # Sample deletion
    # -----------------------------------------------------------------------

    @staticmethod
    def _delete_sample_emcal(d: dict, label: str) -> None:
        """
        Remove sample *label* in-place from an emcal-style result dict.

        Deletes the matching row of ``label`` / ``data``, pops the entry from
        every label-keyed sub-dict (``emiss``, ``rad``, ``sbm`` …; blackbody
        ``bbc`` / ``bbh`` entries are never matched), and drops the aligned row
        from any length-N column of the embedded ``notes`` table.
        """
        labels = [str(s) for s in d.get('label', [])]
        if label not in labels:
            return
        idx = labels.index(label)
        n   = len(labels)

        d['label'] = _delete_axis0(d.get('label'), idx, n)
        if 'data' in d:
            d['data'] = _delete_axis0(d['data'], idx, n)

        for key in _EMCAL_SAMPLE_DICTS:
            sub = d.get(key)
            if isinstance(sub, dict):
                sub.pop(label, None)

        notes = d.get('notes')
        if isinstance(notes, dict):
            for k, v in notes.items():
                notes[k] = _delete_axis0(v, idx, n)

    @staticmethod
    def _delete_sample_sma(d: dict, label: str) -> None:
        """
        Remove sample *label* in-place from an sma-style result dict.

        Every per-sample array (top-level and inside ``grouped``) has the
        matching axis-0 slice removed; endmember-axis and shared arrays are
        left untouched by the length guard in :func:`_delete_axis0`.
        """
        labels = [str(s) for s in d.get('sample_labels', [])]
        if label not in labels:
            return
        idx = labels.index(label)
        n   = len(labels)

        for key in _SMA_SAMPLE_KEYS:
            if key in d:
                d[key] = _delete_axis0(d[key], idx, n)

        gp = d.get('grouped')
        if isinstance(gp, dict):
            for key in _SMA_GROUPED_SAMPLE_KEYS:
                if key in gp:
                    gp[key] = _delete_axis0(gp[key], idx, n)

    @staticmethod
    def _delete_sample_raw(raw: dict, label: str) -> None:
        """Remove sample *label* in-place from a reconstructed raw-data dict."""
        labels = [str(s) for s in raw.get('labels', [])]
        if label not in labels:
            return
        idx = labels.index(label)
        n   = len(labels)

        raw['labels'] = _delete_axis0(raw.get('labels'), idx, n)
        for key in ('sbm', 'radiance', 'emissivity'):
            sub = raw.get(key)
            if isinstance(sub, dict):
                sub.pop(label, None)
        notes = raw.get('notes')
        if notes is not None and hasattr(notes, 'loc'):
            try:
                raw['notes'] = notes[notes['sample_name'] != label].reset_index(drop=True)
            except Exception:
                logging.warning("Could not drop '%s' from raw notes table", label)

    def _on_delete_sample(self, label: str | None) -> None:
        """Delete a spectrum from every loaded in-memory result and refresh the GUI."""
        if not label:
            return

        in_emcal = (self._emcal_result is not None
                    and label in [str(s) for s in self._emcal_result.get('label', [])])
        in_sma   = (self._sma_result is not None
                    and label in [str(s) for s in self._sma_result.get('sample_labels', [])])
        if not (in_emcal or in_sma):
            return

        if not messagebox.askyesno(
            'Delete spectrum',
            f'Remove "{label}" from the loaded results?\n\n'
            'This edits the in-memory results only — the source file on disk is '
            'not changed. Use "Save Results" to write a reduced copy.',
            icon='warning',
        ):
            return

        # Record the position so we can reselect a neighbour after repopulating.
        ref_labels = (self._emcal_result.get('label', []) if in_emcal
                      else self._sma_result.get('sample_labels', []))
        old_idx = [str(s) for s in ref_labels].index(label)

        if in_emcal:
            self._delete_sample_emcal(self._emcal_result, label)
        if in_sma:
            self._delete_sample_sma(self._sma_result, label)
        if self._raw_data is not None:
            self._delete_sample_raw(self._raw_data, label)

        self._data_label = None
        self._sma_label  = None
        logging.info("Deleted sample '%s'", label)

        if self._emcal_result is not None or self._raw_data is not None:
            self._populate_data_tab()
            self._reselect_listbox(self._data_listbox, self._data_labels(),
                                   old_idx, self._on_data_select)
        if self._sma_result is not None:
            self._populate_analysis_tab()
            self._reselect_listbox(self._sma_listbox,
                                   self._sma_result.get('sample_labels', []),
                                   old_idx, self._an_on_select)
        self._refresh_toolbar()

    @staticmethod
    def _reselect_listbox(listbox: tk.Listbox, labels: list,
                          idx: int, on_select) -> None:
        """Select *idx* (clamped) in *listbox* and fire its selection callback."""
        if not labels:
            return
        new_idx = min(idx, len(labels) - 1)
        listbox.selection_clear(0, tk.END)
        listbox.selection_set(new_idx)
        listbox.see(new_idx)
        on_select()

    def _load_folder_raw(self, fdir: str) -> None:
        """Load single-beam spectra from a folder; display immediately without running emcal."""
        errors: list[str] = []

        bbc_terms = ["bbcold", "bbc", "coldbb", "bbwarm", "bbw", "warmbb"]
        bbh_terms = ["bbhot", "bbh", "hotbb"]

        bbc_flist = findFiles(bbc_terms, '.csv', fdir)
        bbh_flist = findFiles(bbh_terms, '.csv', fdir)

        bbc_data: np.ndarray | None = None
        bbh_data: np.ndarray | None = None
        bbc_fname = bbh_fname = ''
        wn: np.ndarray | None = None

        if len(bbc_flist) == 1:
            try:
                bb = readOMNIC(bbc_flist[0])
                bbc_data  = bb['data']
                wn        = bb['wn']
                bbc_fname = Path(bbc_flist[0]).name
            except Exception as exc:
                errors.append(f'BB warm: {exc}')
        elif len(bbc_flist) > 1:
            errors.append(f'Multiple warm BB files found: {bbc_flist}')

        if len(bbh_flist) == 1:
            try:
                bb = readOMNIC(bbh_flist[0])
                bbh_data  = bb['data']
                if wn is None:
                    wn = bb['wn']
                bbh_fname = Path(bbh_flist[0]).name
            except Exception as exc:
                errors.append(f'BB hot: {exc}')
        elif len(bbh_flist) > 1:
            errors.append(f'Multiple hot BB files found: {bbh_flist}')

        # Notes CSV (best-effort; not required for raw display)
        try:
            notes, note_flist = readEmissionCSVnotes(fdir, return_path=True)
        except Exception:
            notes      = None
            note_flist = []

        previous = findFiles(['emcal', 'results'], '.csv', fdir)
        flist = findFiles("", '.csv', fdir)
        flist = [f for f in flist
                 if f not in bbc_flist + bbh_flist + note_flist + previous]
        flist.sort()

        sbm: dict[str, np.ndarray] = {}
        labels: list[str] = []
        for fname in flist:
            try:
                s = readOMNIC(fname)
                lbl = Path(fname).stem
                sbm[lbl] = s['data']
                if wn is None:
                    wn = s['wn']
                labels.append(lbl)
            except Exception as exc:
                errors.append(f'{Path(fname).name}: {exc}')

        if bbc_data is not None:
            sbm['bbc'] = bbc_data
        if bbh_data is not None:
            sbm['bbh'] = bbh_data

        if wn is None:
            self._data_ax.cla()
            self._data_ax.text(0.5, 0.5, f'No spectra found in\n{Path(fdir).name}',
                               ha='center', va='center',
                               transform=self._data_ax.transAxes, color='gray', fontsize=11)
            self._data_canvas.draw_idle()
            logging.warning("No spectra found in %s", fdir)
            return

        self._raw_data = {
            'wn': wn, 'sbm': sbm, 'labels': labels,
            'bbc_fname': bbc_fname, 'bbh_fname': bbh_fname,
            'notes': notes,
        }
        if errors:
            logging.warning("Folder load warnings: %s", '; '.join(errors))
        logging.info("Loaded %d samples from %s (raw SBM)", len(labels), fdir)
        self._data_mode_var.set('sbm')
        self._populate_data_tab()

    def _load_folders_raw(self) -> None:
        """Load SBM from all selected subfolders in a background thread."""
        fdirs = list(self._fdirs)

        def _worker() -> None:
            try:
                results = []
                for fdir in fdirs:
                    r = load_sbm(fdir)
                    try:
                        notes_df, _ = readEmissionCSVnotes(fdir, return_path=True)
                        r['notes'] = notes_df
                    except Exception:
                        r['notes'] = None
                    results.append(r)
                self.after(0, self._on_load_folders_raw_done, results)
            except Exception as exc:
                import traceback
                msg = traceback.format_exc()
                logging.error("load_sbm failed:\n%s", msg)
                self.after(0, lambda m=msg: (
                    messagebox.showerror('SBM load failed', m),
                    self._refresh_toolbar(),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_load_folders_raw_done(self, results: list[dict]) -> None:
        """Merge per-folder load_sbm results and display in SBM mode."""
        import pandas as pd

        ref_xaxis = results[0]['xaxis']
        merged_labels:    list[str]           = []
        merged_sbm:       dict[str, np.ndarray] = {}
        folder_bbs:       list[dict]          = []
        label_folder_map: dict[str, int]      = {}
        notes_dfs:        list                = []

        for fold_idx, r in enumerate(results):
            folder_name = Path(r['fdir']).name
            xaxis = r['xaxis']
            if len(xaxis) != len(ref_xaxis) or not np.all(xaxis == ref_xaxis):
                logging.warning("xaxis mismatch in %s — resampling for display", folder_name)
                sbm_this = {lbl: resample_spectrum(xaxis, arr, ref_xaxis)
                            for lbl, arr in r['sbm'].items()}
            else:
                sbm_this = r['sbm']

            # Collect BBs for this folder (keys may not exist if files were absent)
            folder_bbs.append({k: sbm_this[k] for k in ('bbh', 'bbc') if k in sbm_this})

            label_remap: dict[str, str] = {}
            for lbl in r['label']:
                key = f"{lbl} ({folder_name})" if lbl in merged_sbm else lbl
                merged_labels.append(key)
                merged_sbm[key]      = sbm_this[lbl]
                label_folder_map[key] = fold_idx
                label_remap[lbl]     = key

            notes_df = r.get('notes')
            if notes_df is not None:
                notes_copy = notes_df.copy()
                # Rename sample rows to match any disambiguated labels; leave BB rows unchanged
                notes_copy['sample_name'] = notes_copy['sample_name'].map(
                    lambda n: label_remap.get(n, n)
                )
                notes_dfs.append(notes_copy)

        self._folder_sbm_bbs   = folder_bbs
        self._label_folder_map = label_folder_map

        merged_notes = pd.concat(notes_dfs, ignore_index=True) if notes_dfs else None

        self._raw_data = {
            'wn':     ref_xaxis,
            'sbm':    merged_sbm,
            'labels': merged_labels,
            'bbc_fname': '',
            'bbh_fname': '',
            'notes':  merged_notes,
        }
        logging.info("Loaded %d samples from %d folders (raw SBM)",
                     len(merged_labels), len(results))
        self._data_mode_var.set('sbm')
        self._data_mode = 'sbm'
        self._populate_data_tab()
        self._refresh_toolbar()

    def _on_load_library(self) -> None:
        lib_dir = get_config().get('spectral_libraries_dir') or str(Path(__file__).parent / 'spectral_libraries')
        path = filedialog.askopenfilename(
            title='Load spectral library',
            initialdir=lib_dir,
            filetypes=[('HDF5 files', '*.hdf *.h5 *.hdf5'), ('All files', '*.*')],
        )
        if not path:
            return
        try:
            album = _load_hdf(path)
            self._full_library  = album
            self._browse_source = album
            self._sl_lib_path   = path
            self._sl_populate_listbox()
            self._sl_update_status()
            self._refresh_toolbar()
            self._notebook.select(self._tab_speclib)
            logging.info("Library loaded: %s (%d spectra)", path, len(album))
        except Exception as exc:
            messagebox.showerror('Load error', str(exc))

    def _on_build_library(self) -> None:
        """Launch SpeclibViewer (LWIR mode) as a modal window; on close, offer to load the export."""
        viewer = SpeclibViewer(default_mode='LWIR')
        viewer.transient(self)
        viewer.grab_set()
        viewer.title('SpeclibViewer — Build Library (LWIR)')
        self.wait_window(viewer)

        # After the viewer closes, prompt to load the exported file
        ans = messagebox.askyesno(
            'Load exported library',
            'Do you want to load an exported library from SpeclibViewer?',
        )
        if ans:
            self._on_load_library()

    def _on_run_cal_rad(self) -> None:
        if not self._fdir and not self._fdirs:
            messagebox.showwarning('No data', 'Load a data folder first.')
            return
        dlg = CalRadOptionsDialog(self, self._calrad_opts)
        if dlg.result is None:
            return
        self._calrad_opts.update(dlg.result)
        opts = dlg.result

        if self._fdirs:
            bb_provider = self._make_bb_provider()
            if bb_provider is None:
                return
            self._run_cal_rad_multi(opts, bb_provider)
        else:
            self._run_cal_rad_single(opts)

    def _make_bb_provider(self) -> 'Callable[[], tuple[float, float]] | None':
        """
        Return a BB-temp callback that shows BBTempsDialog.

        Used by the multi-folder cal_rad path where dialogs cannot be shown
        inside the worker thread.  Pre-checks whether a dialog is needed
        (info file missing in the first folder) and shows it upfront.
        Returns None if the user cancels.
        """
        first_fdir = self._fdirs[0] if self._fdirs else self._fdir
        try:
            readEmissionCSVnotes(first_fdir)
            # Notes file found — no callback needed; return a no-op.
            return lambda: (None, None)
        except IOError:
            dlg = BBTempsDialog(self)
            if dlg.result is None:
                return None
            bb1_k, bb2_k = dlg.result
            return lambda: (bb1_k, bb2_k)

    def _run_cal_rad_single(self, opts: dict) -> None:

        def _bb_provider() -> tuple[float, float]:
            dlg = BBTempsDialog(self)
            if dlg.result is None:
                raise MissingTempsError("BB temperature entry canceled.")
            return dlg.result

        try:
            result = cal_rad(
                self._fdir,
                lab                 = opts['lab'],
                bb_emiss            = opts['bb_emiss'],
                noise_free          = opts['noise_free'],
                on_missing_bb_temps = _bb_provider,
            )
            result.pop('_fir',  None)
            result.pop('_data', None)
        except MissingTempsError:
            return
        except Exception as exc:
            messagebox.showerror('cal_rad() failed', str(exc))
            logging.exception("cal_rad() failed")
            return

        self._emcal_result = result
        self._clear_analysis()
        self._data_mode_var.set('radiance')
        self._data_mode = 'radiance'
        self._populate_data_tab()
        self._refresh_toolbar()
        logging.info("cal_rad() complete — %d samples", len(result.get('label', [])))

    def _run_cal_rad_multi(self, opts: dict,
                          bb_provider: 'Callable[[], tuple[float, float]] | None' = None) -> None:
        fdirs       = list(self._fdirs)
        n_total     = len(fdirs)
        parent_name = Path(fdirs[0]).parent.name

        self._btn_calrad['state'] = 'disabled'
        self._btn_emcal['state']  = 'disabled'
        self._btn_sma['state']    = 'disabled'

        def _worker() -> None:
            try:
                results = []
                for i, fdir in enumerate(fdirs):
                    folder_name = Path(fdir).name
                    self.after(0, self._status_var.set,
                               f'{parent_name}  |  cal_rad {i + 1}/{n_total}: {folder_name} ...')
                    r = cal_rad(
                        fdir,
                        lab                 = opts['lab'],
                        bb_emiss            = opts['bb_emiss'],
                        noise_free          = opts['noise_free'],
                        on_missing_bb_temps = bb_provider,
                    )
                    r.pop('_fir',  None)
                    r.pop('_data', None)
                    logging.info("  cal_rad %s — %d sample(s)", folder_name,
                                 len(r.get('label', [])))
                    results.append(r)

                self.after(0, self._status_var.set,
                           f'{parent_name}  |  Merging {n_total} results ...')
                merged = merge(*results, resample=True)
                logging.info("merge() complete — %d samples total",
                             len(merged.get('label', [])))
                self.after(0, self._on_cal_rad_done, results, merged)

            except Exception as exc:
                import traceback
                msg = traceback.format_exc()
                logging.error("cal_rad/merge worker failed:\n%s", msg)
                self.after(0, lambda m=msg: (
                    messagebox.showerror('cal_rad failed', m),
                    self._refresh_toolbar(),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_cal_rad_done(self, individual_results: list[dict], merged: dict) -> None:
        try:
            self._individual_calrad_results = individual_results
            self._emcal_result = merged
            self._clear_analysis()
            self._data_mode_var.set('radiance')
            self._data_mode = 'radiance'
            self._populate_data_tab()
        except Exception as exc:
            logging.exception("_on_cal_rad_done failed")
            messagebox.showerror('Load failed', str(exc))
        finally:
            self._refresh_toolbar()

    def _on_run_emcal(self) -> None:
        if not self._fdir and not self._fdirs:
            messagebox.showwarning('No data', 'Load a data folder first.')
            return
        dlg = EmcalOptionsDialog(self, self._emcal_opts)
        if dlg.result is None:
            return
        self._emcal_opts.update(dlg.result)
        opts = dlg.result

        downwelling_temps: dict[str, float] | None = None
        if opts['lab'] == 'nau' and self._fdir:
            try:
                readEmissionCSVnotes(self._fdir)
            except IOError:
                labels = scan_sample_labels(self._fdir)
                if labels:
                    tdlg = DownwellingTempsDialog(self, labels)
                    if tdlg.result is None:
                        return
                    downwelling_temps = tdlg.result

        if self._fdirs:
            # Pre-collect temps for folders missing the info file before threading.
            bb_provider = None
            downwelling_map: dict[str, dict[str, float]] = {}

            if opts['lab'] == 'nau':
                missing = []
                for fdir in self._fdirs:
                    try:
                        readEmissionCSVnotes(fdir)
                    except IOError:
                        missing.append(fdir)

                if missing:
                    bb_dlg = BBTempsDialog(self)
                    if bb_dlg.result is None:
                        return
                    _bb1_k, _bb2_k = bb_dlg.result
                    bb_provider = lambda: (_bb1_k, _bb2_k)

                    for fdir in missing:
                        folder_name = Path(fdir).name
                        labels = scan_sample_labels(fdir)
                        if labels:
                            tdlg = DownwellingTempsDialog(self, labels,
                                                          folder_name=folder_name)
                            if tdlg.result is None:
                                return
                            downwelling_map[fdir] = tdlg.result

            self._run_emcal_multi(opts, bb_provider=bb_provider,
                                  downwelling_map=downwelling_map)
        else:
            self._run_emcal_single(opts, downwelling_temps)

    def _run_emcal_single(self, opts: dict,
                          downwelling_temps: dict[str, float] | None = None) -> None:

        def _bb_provider() -> tuple[float, float]:
            dlg = BBTempsDialog(self)
            if dlg.result is None:
                raise MissingTempsError("BB temperature entry canceled.")
            return dlg.result

        try:
            result = emcal(
                self._fdir,
                method              = opts['method'],
                lab                 = opts['lab'],
                max_emiss           = opts['max_emiss'],
                bb_emiss            = opts['bb_emiss'],
                n_bb                = opts['n_bb'],
                temp_halfwidth      = opts['temp_halfwidth'],
                violation_weight    = opts['violation_weight'],
                violation_tol       = opts['violation_tol'],
                escalation_factor   = opts['escalation_factor'],
                max_escalations     = opts['max_escalations'],
                noise_free          = opts['noise_free'],
                apply_dehyd         = opts['apply_dehyd'],
                wn_range            = opts['wn_range'],
                downwelling_temps   = downwelling_temps,
                on_missing_bb_temps = _bb_provider,
            )
        except MissingTempsError:
            return  # user canceled a required dialog — abort silently
        except Exception as exc:
            messagebox.showerror('emcal() failed', str(exc))
            logging.exception("emcal() failed")
            return

        self._emcal_result = result
        self._clear_analysis()
        self._data_mode_var.set('emissivity')
        self._data_mode = 'emissivity'
        self._populate_data_tab()
        self._refresh_toolbar()
        logging.info("emcal() complete — %d samples", len(result.get('label', [])))

    def _run_emcal_multi(self, opts: dict,
                         bb_provider=None,
                         downwelling_map: 'dict[str, dict[str, float]] | None' = None) -> None:
        fdirs       = list(self._fdirs)
        n_total     = len(fdirs)
        parent_name = Path(fdirs[0]).parent.name
        _dw_map     = downwelling_map or {}

        self._btn_emcal['state'] = 'disabled'
        self._btn_sma['state']   = 'disabled'

        def _worker() -> None:
            try:
                results = []
                for i, fdir in enumerate(fdirs):
                    folder_name = Path(fdir).name
                    self.after(0, self._status_var.set,
                               f'{parent_name}  |  Processing {i + 1}/{n_total}: {folder_name} ...')
                    r = emcal(
                        fdir,
                        method              = opts['method'],
                        lab                 = opts['lab'],
                        max_emiss           = opts['max_emiss'],
                        bb_emiss            = opts['bb_emiss'],
                        n_bb                = opts['n_bb'],
                        temp_halfwidth      = opts['temp_halfwidth'],
                        violation_weight    = opts['violation_weight'],
                        violation_tol       = opts['violation_tol'],
                        escalation_factor   = opts['escalation_factor'],
                        max_escalations     = opts['max_escalations'],
                        noise_free          = opts['noise_free'],
                        apply_dehyd         = opts['apply_dehyd'],
                        wn_range            = opts['wn_range'],
                        downwelling_temps   = _dw_map.get(fdir),
                        on_missing_bb_temps = bb_provider,
                    )
                    n = len(r.get('sample_labels', r.get('label', [])))
                    logging.info("  emcal %s — %d sample(s)", folder_name, n)
                    results.append(r)

                self.after(0, self._status_var.set,
                           f'{parent_name}  |  Merging {n_total} results ...')
                merged = merge(*results, resample=True)
                n_merged = len(merged.get('sample_labels', merged.get('label', [])))
                logging.info("merge() complete — %d samples total", n_merged)
                self.after(0, self._on_emcal_multi_done, results, merged)

            except Exception as exc:
                import traceback
                msg = traceback.format_exc()
                logging.error("emcal/merge worker failed:\n%s", msg)
                self.after(0, lambda m=msg: (
                    messagebox.showerror('emcal/merge failed', m),
                    self._refresh_toolbar(),
                ))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_emcal_multi_done(self, individual_results: list[dict], merged: dict) -> None:
        try:
            self._individual_emcal_results = individual_results
            self._emcal_result = merged
            self._clear_analysis()
            self._data_mode_var.set('emissivity')
            self._data_mode = 'emissivity'
            self._populate_data_tab()
        except Exception as exc:
            logging.exception("_on_emcal_multi_done failed")
            messagebox.showerror('Load failed', str(exc))
        finally:
            self._refresh_toolbar()

    @staticmethod
    def _sanitize_suffix(raw: str) -> str:
        """
        Clean a user-entered filename suffix, or return '' if unusable.

        Strips surrounding whitespace, replaces path separators with '_' so the
        suffix cannot escape the target folder, and drops leading dots so it
        cannot produce a hidden file. An empty result signals "use a timestamp".
        """
        cleaned = raw.strip().replace(os.sep, '_')
        if os.altsep:
            cleaned = cleaned.replace(os.altsep, '_')
        return cleaned.lstrip('.').strip()

    def _on_save_results(self) -> None:
        # Determine parent output directory
        if self._fdir:
            out_dir = self._fdir
        elif self._fdirs:
            out_dir = str(Path(self._fdirs[0]).parent)
        else:
            out_dir = filedialog.askdirectory(
                title='Choose folder to save results',
                initialdir=self._last_load_dir or os.getcwd())
            if not out_dir:
                return

        # Filename suffix: user-entered value if present, else a timestamp.
        suffix = (self._sanitize_suffix(self._suffix_var.get())
                  or datetime.now().strftime('%Y%m%d_%H%M%S'))
        saved:  list[str] = []
        errors: list[str] = []

        def _save_emcal(result: dict, folder: str, label: str = '') -> None:
            tag = f' ({label})' if label else ''
            hdf_path = os.path.join(folder, f'emcal_results_{suffix}.hdf')
            csv_path = hdf_path.replace('.hdf', '.csv')
            try:
                saveHDF(result, hdf_path)
                saved.append(hdf_path)
                logging.info("Saved emcal result%s → %s", tag, hdf_path)
            except Exception as exc:
                errors.append(f'emcal HDF{tag}: {exc}')
                logging.exception("Failed to save emcal HDF%s", tag)
            try:
                save_emcal_csv(result, csv_path)
                saved.append(csv_path)
                logging.info("Saved emcal CSV%s → %s", tag, csv_path)
            except Exception as exc:
                errors.append(f'emcal CSV{tag}: {exc}')
                logging.exception("Failed to save emcal CSV%s", tag)

        def _save_sma(result: dict, folder: str) -> None:
            hdf_path = os.path.join(folder, f'sma_results_{suffix}.hdf')
            csv_path = hdf_path.replace('.hdf', '.csv')
            try:
                saveHDF(result, hdf_path)
                saved.append(hdf_path)
                logging.info("Saved sma result → %s", hdf_path)
            except Exception as exc:
                errors.append(f'sma HDF: {exc}')
                logging.exception("Failed to save sma HDF")
            try:
                save_sma_csv(result, csv_path)
                saved.append(csv_path)
                logging.info("Saved sma CSV → %s", csv_path)
            except Exception as exc:
                errors.append(f'sma CSV: {exc}')
                logging.exception("Failed to save sma CSV")

        # ── Per-folder individual emcal results ──────────────────────────────
        if self._fdirs and self._individual_emcal_results:
            for fdir, ind_result in zip(self._fdirs, self._individual_emcal_results):
                folder_name = Path(fdir).name
                _save_emcal(ind_result, fdir, label=folder_name)

        # ── Per-folder individual cal_rad results ────────────────────────────
        if self._fdirs and self._individual_calrad_results:
            for fdir, ind_result in zip(self._fdirs, self._individual_calrad_results):
                folder_name = Path(fdir).name
                hdf_path = os.path.join(fdir, f'calrad_results_{suffix}.hdf')
                try:
                    saveHDF(ind_result, hdf_path)
                    saved.append(hdf_path)
                    logging.info("Saved cal_rad result (%s) → %s", folder_name, hdf_path)
                except Exception as exc:
                    errors.append(f'cal_rad HDF ({folder_name}): {exc}')
                    logging.exception("Failed to save cal_rad HDF (%s)", folder_name)

        # ── Merged / single emcal result ─────────────────────────────────────
        if self._emcal_result is not None:
            label = 'merged' if (self._fdirs and self._individual_emcal_results) else ''
            _save_emcal(self._emcal_result, out_dir, label=label)

        # ── SMA result (always on merged data, save to parent) ───────────────
        if self._sma_result is not None:
            _save_sma(self._sma_result, out_dir)

        if saved:
            # Group by directory for a more readable summary
            by_dir: dict[str, list[str]] = {}
            for path in saved:
                d = str(Path(path).parent)
                by_dir.setdefault(d, []).append(Path(path).name)
            lines = []
            for d, names in by_dir.items():
                lines.append(f'{d}:')
                lines.extend(f'  {n}' for n in names)
            messagebox.showinfo('Results saved', '\n'.join(lines))
        if errors:
            messagebox.showerror('Save errors', '\n'.join(errors))

    def _on_run_sma(self) -> None:
        if not self._emcal_result:
            messagebox.showwarning('No emcal result', 'Run emcal() first.')
            return
        if not self._full_library and not self._extra_libs:
            messagebox.showwarning('No library', 'Load a spectral library first.')
            return

        dlg = SmaOptionsDialog(self, self._sma_opts)
        if dlg.result is None:
            return
        self._sma_opts.update(dlg.result)
        opts = dlg.result

        endlib   = self._build_endlib()
        forcedlib = self._build_forcedlib()

        try:
            result = sma(
                self._emcal_result,
                endlib    = endlib,
                forcedlib = forcedlib if forcedlib else None,
                wn_range  = opts['wn_range'],
                bb        = opts['bb'],
                group     = opts['group'],
                nn        = opts['nn'],
                calc_errors = opts['calc_errors'],
                notchco2  = opts['notchco2'],
                slope     = opts['slope'],
                sample_t  = opts['sample_t'],
            )
        except Exception as exc:
            messagebox.showerror('sma() failed', str(exc))
            logging.exception("sma() failed")
            return

        self._sma_result = result
        self._populate_analysis_tab()
        self._refresh_toolbar()
        self._notebook.select(self._tab_analysis)
        logging.info("sma() complete")

    # -----------------------------------------------------------------------
    # Data tab methods
    # -----------------------------------------------------------------------

    def _populate_data_tab(self) -> None:
        r   = self._emcal_result
        raw = self._raw_data
        if r is not None:
            labels = r.get('label', [])
        elif raw is not None:
            labels = raw.get('labels', [])
        else:
            return
        self._data_page = 0
        self._data_listbox.delete(0, tk.END)
        for lbl in labels:
            self._data_listbox.insert(tk.END, lbl)
        if labels:
            self._data_listbox.selection_set(0)
            self._data_label = labels[0]
        self._refresh_data_plot()
        self._refresh_info_tree()
        self._update_metrics_button()

    def _on_data_select(self, _event=None) -> None:
        sel = self._data_listbox.curselection()
        if not sel:
            return
        labels = self._data_labels()
        idx = sel[0]
        if idx < len(labels):
            self._data_label = labels[idx]
            self._refresh_data_plot()
            self._refresh_info_tree()

    def _on_data_mode(self) -> None:
        self._data_mode = self._data_mode_var.get()
        self._refresh_toolbar()
        self._refresh_data_plot()
        self._refresh_info_tree()

    def _on_data_plot_mode(self) -> None:
        self._data_page = 0
        self._refresh_data_plot()

    def _on_data_prev(self) -> None:
        if self._data_plot_var.get() == 'stacked':
            if self._data_page > 0:
                self._data_page -= 1
                self._refresh_data_plot()
        else:
            self._data_navigate(-1)

    def _on_data_next(self) -> None:
        if self._data_plot_var.get() == 'stacked':
            labels = self._data_labels()
            max_page = max(0, (len(labels) - 1) // _PAGE_SIZE)
            if self._data_page < max_page:
                self._data_page += 1
                self._refresh_data_plot()
        else:
            self._data_navigate(1)

    def _data_labels(self) -> list[str]:
        if self._emcal_result is not None:
            return self._emcal_result.get('label', [])
        if self._raw_data is not None:
            return self._raw_data.get('labels', [])
        return []

    def _data_navigate(self, delta: int) -> None:
        labels = self._data_labels()
        if not labels or self._data_label is None:
            return
        try:
            idx = labels.index(self._data_label)
        except ValueError:
            return
        new_idx = max(0, min(len(labels) - 1, idx + delta))
        if new_idx != idx:
            self._data_label = labels[new_idx]
            self._data_listbox.selection_clear(0, tk.END)
            self._data_listbox.selection_set(new_idx)
            self._data_listbox.see(new_idx)
            self._refresh_data_plot()
            self._refresh_info_tree()

    def _on_nav_key(self, delta: int) -> None:
        """Window-level Up/Down handler — dispatches to the active tab, skips input widgets."""
        focused = self.focus_get()
        if isinstance(focused, (ttk.Spinbox, ttk.Combobox, tk.Text,
                                 ttk.Entry, tk.Entry)):
            return
        active = self._notebook.select()
        if active == str(self._tab_data):
            self._data_navigate(delta)
        elif active == str(self._tab_speclib):
            self._sl_navigate(delta)
        elif active == str(self._tab_analysis):
            self._on_an_prev() if delta < 0 else self._on_an_next()

    def _update_data_nav_label(self) -> None:
        labels  = self._data_labels()
        n_total = len(labels)
        if self._data_plot_var.get() == 'individual':
            text = self._data_label or ''
        else:
            start = self._data_page * _PAGE_SIZE + 1
            end   = min(start + _PAGE_SIZE - 1, n_total)
            text  = f'Spectra {start}–{end} of {n_total}' if n_total else ''
        self._data_nav_label.set(text)

    def _refresh_data_plot(self) -> None:
        if self._data_secax is not None:
            try:
                self._data_secax.remove()
            except Exception:
                pass
            self._data_secax = None

        r   = self._emcal_result
        raw = self._raw_data
        ax  = self._data_ax
        ax.cla()

        if r is None and raw is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color='gray')
            self._data_canvas.draw_idle()
            self._update_data_nav_label()
            return

        mode    = self._data_mode_var.get()
        self._data_mode = mode

        # Resolve wavenumber axis, spectra dict, and label list for the active mode
        if mode == 'sbm':
            src     = raw if raw is not None else {}
            wn      = src.get('wn', np.array([]))
            labels  = src.get('labels', [])
            spectra = src.get('sbm', {})
            ax.set_ylabel('Single-beam (counts)')
        elif mode == 'radiance':
            wn     = r.get('xaxis', np.array([]))
            labels = r.get('label', [])
            if self._rad_display_var.get() == 'corrected':
                raw_rad    = r.get('rad', {})
                emiss      = r.get('emiss', {})
                emiss_full = r.get('emiss_full', {})
                spectra = {}
                for lbl in r.get('label', []):
                    if lbl not in raw_rad or lbl not in emiss:
                        continue
                    dw = emiss_full.get(lbl, {}).get('downwelling_rad', None)
                    if dw is None:
                        spectra[lbl] = raw_rad[lbl]
                    else:
                        spectra[lbl] = raw_rad[lbl] - (1.0 - emiss[lbl]) * dw
                if not spectra:
                    spectra = r.get('rad', {})
                    ax.set_ylabel('Radiance (mW m⁻² sr⁻¹ cm)')
                else:
                    ax.set_ylabel('Emission radiance ε·B(T)  (mW m⁻² sr⁻¹ cm)')
            else:
                spectra = r.get('rad', {})
                ax.set_ylabel('Radiance (mW m⁻² sr⁻¹ cm)')
        else:  # emissivity
            wn      = r.get('xaxis', np.array([]))
            labels  = r.get('label', [])
            spectra = r.get('emiss', {})
            ax.set_ylabel('Emissivity')

        plot           = self._data_plot_var.get()
        show_model_fit = self._show_model_var.get() and mode == 'radiance' and r is not None
        show_bb        = self._show_bb_var.get() and mode in ('sbm', 'radiance')

        def _plot_bbs(lw: float) -> None:
            if not show_bb:
                return
            bb_pairs = [
                ('bbh', dict(color='red',  ls='--', lw=lw, label='BB hot')),
                ('bbc', dict(color='blue', ls=':',  lw=lw, label='BB warm')),
            ]
            if mode == 'sbm' and self._folder_sbm_bbs:
                # Multi-folder: resolve per-folder BB source(s)
                if plot == 'stacked':
                    labeled: set[str] = set()
                    for fold_bb in self._folder_sbm_bbs:
                        for key, style in bb_pairs:
                            d = fold_bb.get(key)
                            if d is None:
                                continue
                            s = dict(style)
                            if key in labeled:
                                s.pop('label', None)
                            else:
                                labeled.add(key)
                            ax.plot(wn, d, **s)
                else:
                    fold_idx = self._label_folder_map.get(self._data_label, 0)
                    fold_bb  = (self._folder_sbm_bbs[fold_idx]
                                if fold_idx < len(self._folder_sbm_bbs) else {})
                    for key, style in bb_pairs:
                        d = fold_bb.get(key)
                        if d is not None:
                            ax.plot(wn, d, **style)
            else:
                bb_src = (spectra if mode == 'sbm' else r.get('rad', {})) if r or mode == 'sbm' else {}
                for key, style in bb_pairs:
                    d = bb_src.get(key)
                    if d is not None:
                        ax.plot(wn, d, **style)
        colors         = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        def _model_overlay(lbl: str):
            """Return rad0 adjusted to match the active radiance display space."""
            rad0 = r.get('rad0', {}).get(lbl)
            if rad0 is None:
                return None
            if self._rad_display_var.get() == 'corrected':
                dw = r.get('emiss_full', {}).get(lbl, {}).get('downwelling_rad')
                if dw is not None:
                    max_emiss = r.get('max_emiss', 0.98)
                    rad0 = rad0 - (1.0 - max_emiss) * dw
            return rad0

        if plot == 'stacked':
            start     = self._data_page * _PAGE_SIZE
            page_lbls = labels[start:start + _PAGE_SIZE]
            for i, lbl in enumerate(page_lbls):
                col = colors[i % len(colors)]
                if lbl in spectra:
                    ax.plot(wn, spectra[lbl], label=lbl, color=col, lw=0.8)
                if show_model_fit:
                    rad0 = _model_overlay(lbl)
                    if rad0 is not None:
                        ax.plot(wn, rad0, color=col, ls='--', lw=0.7, alpha=0.6)
            _plot_bbs(lw=1.2)
            ax.legend(fontsize=7, loc='upper right')
        else:
            lbl = self._data_label
            if lbl and lbl in spectra:
                (line,) = ax.plot(wn, spectra[lbl], lw=1.2)
                if show_model_fit:
                    rad0 = _model_overlay(lbl)
                    if rad0 is not None:
                        ax.plot(wn, rad0, color=line.get_color(),
                                ls='--', lw=1.0, label='Model fit')
                        ax.legend(fontsize=9)
                _plot_bbs(lw=1.0)
                if show_bb or show_model_fit:
                    ax.legend(fontsize=9)
                ax.set_title(lbl, fontsize=10)

        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.invert_xaxis()
        self._data_secax = _add_top_axis(ax)
        self._data_canvas.draw_idle()
        self._update_data_nav_label()

    def _ensure_notes(self) -> bool:
        """Load and cache the notes DataFrame. Returns True if available."""
        if self._notes_df is not None:
            return True
        fdir = self._fdir or (self._fdirs[0] if self._fdirs else None)
        if fdir is None:
            return False
        try:
            self._notes_df, _ = readEmissionCSVnotes(fdir, return_path=True)
            return True
        except Exception:
            return False

    def _get_notes_row(self, label: str) -> dict:
        """Return the notes DataFrame row for *label* as a plain dict, or {}."""
        if not self._ensure_notes():
            return {}
        match = self._notes_df[self._notes_df['sample_name'] == label]
        if len(match) == 0:
            return {}
        return match.iloc[0].to_dict()

    def _get_bb_resistances(self) -> dict:
        """Return {bbhot: (ch1, ch2), bbwarm: (ch1, ch2)} from the notes, or {}."""
        if not self._ensure_notes():
            return {}
        result = {}
        for bb_name in ('bbhot', 'bbwarm'):
            match = self._notes_df[self._notes_df['sample_name'] == bb_name]
            if len(match) > 0:
                row = match.iloc[0]
                ch1 = row.get('channel_101', float('nan'))
                ch2 = row.get('channel_102', float('nan'))
                try:
                    result[bb_name] = (float(ch1), float(ch2))
                except (TypeError, ValueError):
                    pass
        return result

    def _get_notes_all(self) -> tuple[list, dict | None]:
        """
        Return (sample_labels, notes_dict) for the instrument metrics plot.

        Priority:
        1. ``_emcal_result['notes']`` — embedded by cal_rad, column-keyed dict of lists.
        2. ``_raw_data['notes']`` — pandas DataFrame loaded during SBM-only folder load.
        3. ``_notes_df`` via ``_ensure_notes()`` — on-demand disk read.

        Returns ([], None) when no notes data is available.
        """
        import pandas as pd

        _ch_cols = ('channel_103', 'channel_104', 'channel_105', 'channel_106', 'channel_107')

        def _dt_str(val) -> str:
            if pd.isna(val):
                return ''
            return val.isoformat() if hasattr(val, 'isoformat') else str(val)

        def _df_to_notes(df: 'pd.DataFrame', labels: list) -> dict:
            notes: dict = {'dtime': [], **{c: [] for c in _ch_cols}}
            for lbl in labels:
                row = df[df['sample_name'] == lbl]
                if len(row) > 0:
                    r0 = row.iloc[0]
                    notes['dtime'].append(_dt_str(r0.get('dtime', '')))
                    for col in _ch_cols:
                        val = r0.get(col, float('nan'))
                        notes[col].append(float(val) if pd.notna(val) else float('nan'))
                else:
                    notes['dtime'].append('')
                    for col in _ch_cols:
                        notes[col].append(float('nan'))
            # BB resistance rows for secondary axis
            bb_names, bb_dtimes, bb_ch101, bb_ch102 = [], [], [], []
            for bb in ('bbwarm', 'bbhot'):
                row = df[df['sample_name'] == bb]
                if len(row) > 0:
                    r0 = row.iloc[0]
                    bb_names.append(bb)
                    bb_dtimes.append(_dt_str(r0.get('dtime', '')))
                    for col, lst in (('channel_101', bb_ch101), ('channel_102', bb_ch102)):
                        val = r0.get(col, float('nan'))
                        lst.append(float(val) if pd.notna(val) else float('nan'))
            notes['bb_name']  = bb_names
            notes['bb_dtime'] = bb_dtimes
            notes['bb_ch101'] = bb_ch101
            notes['bb_ch102'] = bb_ch102
            return notes

        # Priority 1: embedded notes from cal_rad output
        r = self._emcal_result
        if r is not None and 'notes' in r:
            return list(r.get('label', [])), r['notes']

        # Priority 2: DataFrame embedded in _raw_data (single-folder SBM load)
        raw = self._raw_data
        labels: list = []
        if raw is not None:
            labels = list(raw.get('labels', []))
            df = raw.get('notes')
            if df is not None:
                return labels, _df_to_notes(df, labels)

        # Priority 3: on-demand disk read
        if not labels:
            return [], None
        if not self._ensure_notes():
            return labels, None
        return labels, _df_to_notes(self._notes_df, labels)

    def _update_metrics_button(self) -> None:
        """Enable/disable the Instrument Metrics button based on notes availability."""
        labels, notes = self._get_notes_all()
        state = tk.NORMAL if (labels and notes is not None) else tk.DISABLED
        self._btn_instrument_metrics.config(state=state)

    def _on_plot_instrument_metrics(self) -> None:
        """Open (or raise) the InstrumentMetricsDialog."""
        if self._metrics_dialog is not None:
            try:
                self._metrics_dialog.lift()
                self._metrics_dialog.focus_force()
                return
            except tk.TclError:
                self._metrics_dialog = None
        labels, notes = self._get_notes_all()
        if not labels or notes is None:
            return
        self._metrics_dialog = InstrumentMetricsDialog(self, labels, notes)

    def _refresh_info_tree(self) -> None:
        self._clear_info_tree()
        lbl = self._data_label
        r   = self._emcal_result
        if lbl is None:
            return

        tree = self._info_tree

        def _insert_group(name: str, rows: list[tuple[str, str]]) -> None:
            if not rows:
                return
            grp = tree.insert('', tk.END, text='', values=(name, ''),
                              open=True, tags=('header',))
            for i, (field, value) in enumerate(rows):
                tree.insert(grp, tk.END, text='', values=(field, value),
                            tags=('odd' if i % 2 else 'even',))

        # ── Group 1: Measurement Info ────────────────────────────────────────
        meas_rows: list[tuple[str, str]] = []
        note = self._get_notes_row(lbl)
        if note:
            t = note.get('dtime')
            if t is not None:
                meas_rows.append(('Meas. time', str(t)))
            for col, label in [
                (f'channel_{n}', f'{_CHANNEL_LABELS[n]} (°C)') for n in (103, 104, 105, 106, 107)
            ]:
                val = note.get(col)
                try:
                    fval = float(val)
                    if not np.isnan(fval):
                        meas_rows.append((label, f'{fval:.2f}'))
                except (TypeError, ValueError):
                    pass
        bb_res = self._get_bb_resistances()
        for bb_name, display in (('bbhot', 'BB hot'), ('bbwarm', 'BB warm')):
            ch = bb_res.get(bb_name)
            if ch is not None:
                ch1, ch2 = ch
                if not np.isnan(ch1):
                    meas_rows.append((f'{display} — {_CHANNEL_LABELS[101]} (Ω)', f'{ch1:.2f}'))
                if not np.isnan(ch2):
                    meas_rows.append((f'{display} — {_CHANNEL_LABELS[102]} (Ω)', f'{ch2:.2f}'))
        _insert_group('Measurement Info', meas_rows)

        # ── Group 2: Radiance Calibration ────────────────────────────────────
        cal_rows: list[tuple[str, str]] = []
        if r is not None:
            calib = r.get('calib', {})
            bbc = calib.get('bbc_temp')
            bbh = calib.get('bbh_temp')
            if bbc is not None:
                cal_rows.append(('BB warm (K)',    f'{float(bbc):.1f}'))
            if bbh is not None:
                cal_rows.append(('BB hot (K)',     f'{float(bbh):.1f}'))
            nf = calib.get('noise_free')
            if nf is not None:
                cal_rows.append(('Noise-free IRF', str(nf)))
        _insert_group('Radiance Calibration', cal_rows)

        # ── Group 3: Emissivity Retrieval ────────────────────────────────────
        em_rows: list[tuple[str, str]] = []
        if r is not None:
            em_rows.append(('Method',    r.get('method', '—')))
            max_e = r.get('max_emiss')
            if max_e is not None:
                em_rows.append(('max_emiss', f'{float(max_e):.3f}'))
            temp = r.get('sample_temps', {}).get(lbl)
            if temp is not None:
                em_rows.append(('Retrieved T (K)', f'{float(temp):.1f}'))

            em_full = r.get('emiss_full', {}).get(lbl, {})

            dw_t = em_full.get('downwelling_t')
            if dw_t is not None:
                em_rows.append(('Downwelling T (K)', f'{float(dw_t):.1f}'))
            dw_e = em_full.get('downwelling_e')
            if dw_e is not None:
                em_rows.append(('Downwelling ε',     f'{float(dw_e):.4f}'))

            wn_r = em_full.get('wn_range')
            if wn_r is not None:
                em_rows.append(('Wn range (cm⁻¹)', f'{int(wn_r[0])}–{int(wn_r[1])}'))

            # NEM-specific
            wn_t = r.get('sample_t_wavenumber', {}).get(lbl)
            if wn_t is not None and not np.isnan(float(wn_t)):
                em_rows.append(('T wavenumber (cm⁻¹)', f'{float(wn_t):.1f}'))
            for key, label in [
                ('max_t1',           'BT warm-range peak (K)'),
                ('wn_t1',            'BT warm-range wn (cm⁻¹)'),
                ('max_t2',           'BT cold-range peak (K)'),
                ('wn_t2',            'BT cold-range wn (cm⁻¹)'),
                ('threshold_t_warm', 'Threshold T warm (K)'),
                ('threshold_t_cold', 'Threshold T cold (K)'),
                ('toffset',          'T offset (K)'),
            ]:
                val = em_full.get(key)
                if val is not None:
                    em_rows.append((label, f'{float(val):.1f}'))
            co2r = em_full.get('co2_range')
            if co2r is not None:
                em_rows.append(('CO₂ range (cm⁻¹)',    f'{int(co2r[0])}–{int(co2r[1])}'))
            wn_rc = em_full.get('wn_range_cold')
            if wn_rc is not None:
                em_rows.append(('Wn range cold (cm⁻¹)', f'{int(wn_rc[0])}–{int(wn_rc[1])}'))

            # Hullfit-specific
            bb_temps = em_full.get('bb_temps')
            bb_fracs = em_full.get('bb_fracs')
            if bb_temps is not None and bb_fracs is not None:
                bb_temps = list(bb_temps)
                bb_fracs = list(bb_fracs)
                em_rows.append(('n_bb', str(len(bb_temps))))
                for k, (bt, bf) in enumerate(zip(bb_temps, bb_fracs)):
                    em_rows.append((f'BB{k+1} temp (K)', f'{float(bt):.1f}'))
                    em_rows.append((f'BB{k+1} frac',     f'{float(bf):.3f}'))
                weights = np.array(bb_fracs)
                values  = np.array(bb_temps)
                if weights.sum() > 0:
                    em_rows.append(('Weighted mean T (K)',
                                    f'{float(np.average(values, weights=weights)):.1f}'))
            for key, label, fmt in [
                ('r2',      'R²',          '.4f'),
                ('rmse',    'RMSE',        '.4f'),
                ('nfev',    'N iterations', None),
                ('elapsed', 'Elapsed (s)', '.2f'),
            ]:
                val = em_full.get(key)
                if val is not None:
                    try:
                        fval = format(int(float(val))) if fmt is None else format(float(val), fmt)
                        em_rows.append((label, fval))
                    except (TypeError, ValueError):
                        pass
        _insert_group('Emissivity Retrieval', em_rows)

    def _clear_info_tree(self) -> None:
        for item in self._info_tree.get_children():
            self._info_tree.delete(item)

    def _clear_analysis(self) -> None:
        self._sma_result        = None
        self._sma_lib_unit      = {}
        self._sma_lib_raw       = {}
        self._sma_group_lib_raw = {}
        self._sma_label         = None
        self._sma_listbox.delete(0, tk.END)
        self._an_refresh_plot()
        self._an_refresh_table()

    # -----------------------------------------------------------------------
    # Speclib tab methods
    # -----------------------------------------------------------------------


    def _sl_populate_listbox(self) -> None:
        self._sl_page = 0
        self._filtered_ids = list(self._browse_source.keys())
        self._sl_listbox.delete(0, tk.END)
        for sid in self._filtered_ids:
            entry = self._browse_source[sid]
            name  = entry.get('sample_name', str(sid))
            tag   = ''
            if sid in self._excluded_sids:
                tag = _TAG_EXCL
            elif sid in self._forced_sids:
                tag = _TAG_FORCE
            self._sl_listbox.insert(tk.END, name)
            if tag:
                idx = self._sl_listbox.size() - 1
                self._sl_listbox.itemconfigure(idx, **self._sl_tag_kwargs(tag))
        self._sl_update_em_status()

    def _sl_tag_kwargs(self, tag: str) -> dict:
        if tag == _TAG_EXCL:
            return dict(foreground='#c0392b')   # red
        if tag == _TAG_FORCE:
            return dict(foreground='#27ae60')   # green
        return {}

    def _sl_navigate(self, delta: int) -> None:
        lb = self._sl_listbox
        n  = lb.size()
        if n == 0:
            return
        sel = lb.curselection()
        cur = sel[0] if sel else 0
        new_idx = max(0, min(n - 1, cur + delta))
        if new_idx != cur:
            lb.selection_clear(0, tk.END)
            lb.selection_set(new_idx)
            lb.see(new_idx)
            self._sl_on_select()

    def _sl_on_select(self, _event=None) -> None:
        sel = self._sl_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx < len(self._filtered_ids):
            self._active_sid = self._filtered_ids[idx]
            self._sl_refresh_plot()

    def _sl_update_status(self) -> None:
        n = len(self._browse_source)
        if self._sl_lib_path:
            name = Path(self._sl_lib_path).name
            self._sl_status.config(
                text=f'{name}  ({n} spectra)',
                foreground='black')
        else:
            self._sl_status.config(text='No library loaded', foreground='gray')

    def _sl_update_em_status(self) -> None:
        n_excl  = len(self._excluded_sids)
        n_force = len(self._forced_sids)
        total   = len(self._browse_source)
        n_em    = total - n_excl
        parts   = [f'{n_em} endmembers']
        if n_excl:  parts.append(f'{n_excl} excluded')
        if n_force: parts.append(f'{n_force} forced')
        self._sl_em_status.config(text=' | '.join(parts))

    def _sl_toggle_excluded(self) -> None:
        sel = self._sl_listbox.curselection()
        if not sel or sel[0] >= len(self._filtered_ids):
            return
        sid = self._filtered_ids[sel[0]]
        if sid in self._excluded_sids:
            self._excluded_sids.discard(sid)
        else:
            self._excluded_sids.add(sid)
            self._forced_sids.discard(sid)
        self._sl_populate_listbox()

    def _sl_toggle_forced(self) -> None:
        sel = self._sl_listbox.curselection()
        if not sel or sel[0] >= len(self._filtered_ids):
            return
        sid = self._filtered_ids[sel[0]]
        if sid in self._forced_sids:
            self._forced_sids.discard(sid)
        else:
            self._forced_sids.add(sid)
            self._excluded_sids.discard(sid)
        self._sl_populate_listbox()

    def _sl_remove_spectrum(self) -> None:
        sel = self._sl_listbox.curselection()
        if not sel or sel[0] >= len(self._filtered_ids):
            return
        sid = self._filtered_ids[sel[0]]
        for store in (self._full_library, self._current_album, self._extra_libs):
            if sid in store:
                del store[sid]
        self._excluded_sids.discard(sid)
        self._forced_sids.discard(sid)
        self._sl_populate_listbox()
        self._sl_update_status()

    def _on_sl_prev(self) -> None:
        if self._sl_plot_var.get() == 'stacked':
            if self._sl_page > 0:
                self._sl_page -= 1
                self._sl_refresh_plot()
        else:
            self._sl_navigate(-1)

    def _on_sl_next(self) -> None:
        if self._sl_plot_var.get() == 'stacked':
            max_page = max(0, (len(self._filtered_ids) - 1) // _PAGE_SIZE)
            if self._sl_page < max_page:
                self._sl_page += 1
                self._sl_refresh_plot()
        else:
            self._sl_navigate(1)

    def _sl_navigate(self, delta: int) -> None:
        if not self._filtered_ids or self._active_sid is None:
            return
        try:
            idx = self._filtered_ids.index(self._active_sid)
        except ValueError:
            return
        new_idx = max(0, min(len(self._filtered_ids) - 1, idx + delta))
        if new_idx != idx:
            self._active_sid = self._filtered_ids[new_idx]
            self._sl_listbox.selection_clear(0, tk.END)
            self._sl_listbox.selection_set(new_idx)
            self._sl_listbox.see(new_idx)
            self._sl_refresh_plot()

    def _update_sl_nav_label(self) -> None:
        n_total = len(self._filtered_ids)
        if self._sl_plot_var.get() == 'individual':
            if self._active_sid is not None:
                entry = self._browse_source.get(self._active_sid, {})
                text  = entry.get('sample_name', str(self._active_sid))
            else:
                text = ''
        else:
            start = self._sl_page * _PAGE_SIZE + 1
            end   = min(start + _PAGE_SIZE - 1, n_total)
            text  = f'Spectra {start}–{end} of {n_total}' if n_total else ''
        self._sl_nav_label.set(text)

    def _sl_refresh_plot(self) -> None:
        if self._sl_secax is not None:
            try:
                self._sl_secax.remove()
            except Exception:
                pass
            self._sl_secax = None

        ax   = self._sl_ax
        ax.cla()
        mode   = self._sl_plot_var.get()
        colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

        if mode == 'individual' and self._active_sid is not None:
            entry = self._browse_source.get(self._active_sid)
            if entry:
                xaxis = entry.get('xaxis', np.array([]))
                data  = entry.get('data',  np.array([]))
                label = entry.get('sample_name', str(self._active_sid))
                color = '#27ae60' if self._active_sid in self._forced_sids else \
                        '#aaaaaa' if self._active_sid in self._excluded_sids else 'tab:red'
                ax.plot(xaxis, data, color=color, lw=1.2)
                ax.set_title(label, fontsize=10)
        elif mode == 'stacked':
            start = self._sl_page * _PAGE_SIZE
            page_ids = self._filtered_ids[start:start + _PAGE_SIZE]
            for i, sid in enumerate(page_ids):
                entry = self._browse_source.get(sid)
                if entry is None:
                    continue
                xaxis = entry.get('xaxis', np.array([]))
                data  = entry.get('data',  np.array([]))
                label = entry.get('sample_name', str(sid))
                color = '#aaaaaa' if sid in self._excluded_sids else \
                        '#c0392b' if sid in self._forced_sids else colors[i % len(colors)]
                ax.plot(xaxis, data, label=label, color=color, lw=0.8)
            if page_ids:
                ax.legend(fontsize=7, loc='upper right')

        ax.set_xlabel('Wavenumber (cm⁻¹)')
        ax.set_ylabel('Emissivity')
        ax.invert_xaxis()
        self._sl_secax = _add_top_axis(ax)
        self._sl_canvas.draw_idle()
        self._update_sl_nav_label()
        self._sl_refresh_info()

    # Fields shown in individual mode (key, display label)
    _SL_SUPPRESS = {'nan', '', '-', 'none', 'n/a'}

    def _sl_refresh_info(self) -> None:
        tree = self._sl_info_tree
        for item in tree.get_children():
            tree.delete(item)
        mode = self._sl_plot_var.get()
        if mode != 'individual' or self._active_sid is None:
            return
        entry = self._browse_source.get(self._active_sid)
        if not entry:
            return
        for group_name, fields in _INFO_GROUPS:
            collapsed = group_name in _INFO_COLLAPSED_BY_DEFAULT
            grp_iid = tree.insert(
                '', tk.END, text='', values=(group_name, ''),
                open=not collapsed, tags=('header',),
            )
            row_idx = 0
            for field, label in fields:
                v = entry.get(field, '')
                if hasattr(v, 'item'):
                    v = v.item()
                if str(v).strip().lower() in self._SL_SUPPRESS:
                    continue
                tag = 'odd' if row_idx % 2 else 'even'
                tree.insert(grp_iid, tk.END, text='', values=(label, v), tags=(tag,))
                row_idx += 1

    def _build_endlib(self) -> dict:
        """Build the main endlib dict, excluding marked spectra."""
        return {
            sid: entry
            for sid, entry in self._browse_source.items()
            if sid not in self._excluded_sids and sid not in self._forced_sids
        }

    def _build_forcedlib(self) -> dict:
        """Build the forced endmember library from marked spectra."""
        return {
            sid: entry
            for sid, entry in self._browse_source.items()
            if sid in self._forced_sids
        }

    # -----------------------------------------------------------------------
    # Analysis tab methods
    # -----------------------------------------------------------------------

    def _populate_analysis_tab(self) -> None:
        r = self._sma_result
        if r is None:
            return
        labels = r.get('sample_labels', [])
        self._sma_listbox.delete(0, tk.END)
        for lbl in labels:
            self._sma_listbox.insert(tk.END, lbl)
        if labels:
            self._sma_listbox.selection_set(0)
            self._sma_label = labels[0]

        # Pre-compute normalised endmember overlay spectra (shared across samples)
        xaxis    = np.asarray(r.get('xaxis', []))
        endlib   = r.get('endlib') or {}
        em_labels = r.get('labels', [])
        wn_mask  = (xaxis >= 500) & (xaxis <= 1500)
        self._sma_lib_unit: dict[str, np.ndarray] = {}

        self._sma_lib_raw: dict[str, np.ndarray] = {}
        if endlib and len(xaxis) > 0:
            label_specs: dict[str, list] = {lb: [] for lb in em_labels}
            for sid, entry in endlib.items():
                spec = resample_spectrum(entry['xaxis'], entry['data'], xaxis)
                entry_label = (
                    f"{entry.get('sample_name', 'Unknown')} "
                    f"{entry.get('spec_id', sid)}"
                )
                category = str(entry.get('category', ''))
                for lb in em_labels:
                    if entry_label == lb or category == lb:
                        label_specs[lb].append(spec)
                        break
            for lb, specs in label_specs.items():
                if not specs:
                    continue
                mean_spec  = np.mean(specs, axis=0)
                spec_range = mean_spec[wn_mask]
                lo, hi     = float(spec_range.min()), float(spec_range.max())
                self._sma_lib_unit[lb] = (mean_spec - lo) / (hi - lo + 1e-12)
                self._sma_lib_raw[lb]  = mean_spec

        # Pre-compute grouped overlay spectra (category-averaged, for group mode)
        self._sma_group_lib_unit: dict[str, np.ndarray] = {}
        self._sma_group_lib_raw:  dict[str, np.ndarray] = {}
        gp = r.get('grouped')
        if endlib and gp is not None and len(xaxis) > 0:
            group_labels = list(gp.get('grouped_labels', []))
            group_specs: dict[str, list] = {lb: [] for lb in group_labels}
            for sid, entry in endlib.items():
                cat = str(entry.get('category', ''))
                if cat in group_specs:
                    spec = resample_spectrum(entry['xaxis'], entry['data'], xaxis)
                    group_specs[cat].append(spec)
            for lb, specs in group_specs.items():
                if not specs:
                    continue
                mean_spec  = np.mean(specs, axis=0)
                spec_range = mean_spec[wn_mask]
                lo, hi     = float(spec_range.min()), float(spec_range.max())
                self._sma_group_lib_unit[lb] = (mean_spec - lo) / (hi - lo + 1e-12)
                self._sma_group_lib_raw[lb]  = mean_spec

        self._an_refresh_plot()
        self._an_refresh_table()

    def _an_on_select(self, _event=None) -> None:
        sel = self._sma_listbox.curselection()
        if not sel:
            return
        r = self._sma_result
        if r is None:
            return
        labels = r.get('sample_labels', [])
        idx = sel[0]
        if idx < len(labels):
            self._sma_label = labels[idx]
            self._an_refresh_plot()
            self._an_refresh_table()

    def _an_navigate(self, delta: int) -> None:
        r = self._sma_result
        if r is None:
            return
        labels = r.get('sample_labels', [])
        if not labels:
            return
        cur = labels.index(self._sma_label) if self._sma_label in labels else -1
        new_idx = max(0, min(len(labels) - 1, cur + delta))
        if new_idx == cur:
            return
        self._sma_label = labels[new_idx]
        self._sma_listbox.selection_clear(0, tk.END)
        self._sma_listbox.selection_set(new_idx)
        self._sma_listbox.see(new_idx)
        self._an_refresh_plot()
        self._an_refresh_table()

    def _on_an_prev(self) -> None:
        self._an_navigate(-1)

    def _on_an_next(self) -> None:
        self._an_navigate(1)

    def _an_init_sash(self) -> None:
        h = self._an_right_vpaned.winfo_height()
        if h > 10:
            self._an_right_vpaned.sashpos(0, h // 3)

    def _an_refresh_plot(self) -> None:
        fig = self._an_fig
        fig.clear()

        r = self._sma_result
        if r is None or self._sma_label is None:
            self._an_canvas.draw_idle()
            self._an_refresh_pie()
            return

        lbl           = self._sma_label
        sample_labels = r.get('sample_labels', [])
        if lbl not in sample_labels:
            self._an_canvas.draw_idle()
            self._an_refresh_pie()
            return
        idx = sample_labels.index(lbl)

        xaxis         = np.asarray(r.get('xaxis', []))
        measured      = r.get('measured')
        modeled       = r.get('modeled')
        bb_normconc   = r.get('bb_normconc')
        slope_normconc = r.get('slope_normconc')
        rms           = r.get('rms')
        wn_mask       = (xaxis >= 500) & (xaxis <= 1500)
        wn_range      = r.get('wn_range', (float(xaxis.min()), float(xaxis.max())))
        wn_lo, wn_hi  = wn_range

        use_group    = self._an_group_var.get()
        use_residual = self._an_residual_var.get()
        use_error    = self._an_error_var.get()
        use_cumulative = self._an_cumulative_var.get()
        use_other    = self._an_other_var.get()
        try:
            threshold = float(self._an_threshold_var.get())
        except (tk.TclError, ValueError):
            threshold = 5.0
        try:
            offset_val = float(self._an_offset_var.get())
        except (tk.TclError, ValueError):
            offset_val = 0.0

        # Select display data: grouped or individual
        gp = r.get('grouped') if use_group and r.get('grouped') else None
        if gp is not None:
            disp_normconc  = np.atleast_2d(gp['grouped_normconc'])
            disp_labels    = list(gp['grouped_labels'])
            disp_normerror = (np.atleast_2d(gp['grouped_normerror'])
                              if 'grouped_normerror' in gp else None)
            lib_unit = self._sma_group_lib_unit
            lib_raw  = self._sma_group_lib_raw
        else:
            nc = r.get('normconc')
            disp_normconc  = np.atleast_2d(nc) if nc is not None else None
            disp_labels    = r.get('labels', [])
            disp_normerror = (np.atleast_2d(r['normerror'])
                              if 'normerror' in r else None)
            lib_unit = self._sma_lib_unit
            lib_raw  = self._sma_lib_raw

        meas_i  = np.asarray(measured[idx]) if measured is not None else None
        model_i = np.asarray(modeled[idx])  if modeled  is not None else None
        conc_i  = disp_normconc[idx]        if disp_normconc is not None else None
        err_i   = disp_normerror[idx]       if (use_error and disp_normerror is not None) else None

        rms_val    = float(rms[idx]) if rms is not None and idx < len(rms) else float('nan')
        bb_val     = float(np.asarray(bb_normconc)[idx])    if bb_normconc    is not None else 0.0
        sl_val     = float(np.asarray(slope_normconc)[idx]) if slope_normconc is not None else 0.0
        delta_t    = r.get('delta_t_estimated')
        dt_val     = float(np.asarray(delta_t)[idx]) if delta_t is not None else float('nan')
        n_total    = len(sample_labels)
        if slope_normconc is not None:
            _dt_str = f' (ΔT≈{dt_val:.1f} K)' if np.isfinite(dt_val) else ''
            sl_str  = f'  Slope = {sl_val:.1f}%{_dt_str}'
        else:
            sl_str  = ''
        title = f'{lbl}  —  BB = {bb_val:.1f}%{sl_str}  —  RMS = {rms_val:.4f}  [{idx + 1}/{n_total}]'

        def _shade_excluded(ax) -> None:
            xlo, xhi = float(xaxis.min()), float(xaxis.max())
            kw = dict(color='gray', alpha=0.15, zorder=0, lw=0)
            if xlo < wn_lo:
                ax.axvspan(xlo, wn_lo, **kw)
            if wn_hi < xhi:
                ax.axvspan(wn_hi, xhi, **kw)

        # 4-case layout (mirrors plot_sma):
        #   lib_unit + residual  →  3 panels [3, 5, 2]
        #   lib_unit only        →  2 panels [3, 5]
        #   residual only        →  2 panels [5, 2]
        #   neither              →  1 panel
        kw = dict(left=0.10, right=0.95)
        if lib_unit and use_residual:
            gs  = fig.add_gridspec(3, 1, height_ratios=[3, 5, 2],
                                   hspace=0.03, top=0.88, bottom=0.10, **kw)
            ax_top = fig.add_subplot(gs[0])
            ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
            ax_res = fig.add_subplot(gs[2], sharex=ax_top)
        elif lib_unit:
            gs  = fig.add_gridspec(2, 1, height_ratios=[3, 5],
                                   hspace=0.03, top=0.88, bottom=0.10, **kw)
            ax_top = fig.add_subplot(gs[0])
            ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
            ax_res = None
        elif use_residual:
            gs  = fig.add_gridspec(2, 1, height_ratios=[5, 2],
                                   hspace=0.06, top=0.93, bottom=0.10, **kw)
            ax_top = None
            ax_bot = fig.add_subplot(gs[0])
            ax_res = fig.add_subplot(gs[1], sharex=ax_bot)
        else:
            fig.subplots_adjust(top=0.87, bottom=0.10, **kw)
            ax_top = None
            ax_bot = fig.add_subplot(111)
            ax_res = None

        self._an_ax_top   = ax_top
        self._an_ax_main  = ax_bot
        self._an_ax_resid = ax_res

        # Overlay panel — endmember / group spectra
        if ax_top is not None:
            if conc_i is not None:
                order = np.argsort(conc_i)[::-1]
                above = [
                    (j, disp_labels[j] if j < len(disp_labels) else '')
                    for j in order
                    if conc_i[j] >= threshold
                    and (disp_labels[j] if j < len(disp_labels) else '') in lib_raw
                ]

                # Optional Other composite from below-threshold endmembers
                other_item = None
                if use_other:
                    below_frac = 0.0
                    below_wsum = np.zeros_like(xaxis)
                    for j in order:
                        lb = disp_labels[j] if j < len(disp_labels) else ''
                        c  = conc_i[j]
                        if 0.0 < c < threshold and lb in lib_raw:
                            f = c / 100.0
                            below_frac += f
                            below_wsum += lib_raw[lb] * f
                    if below_frac > 0.0:
                        other_item = (below_frac, below_wsum)

                if use_cumulative:
                    stack_top = np.ones_like(xaxis)
                    for rank, (j, lb) in enumerate(above):
                        frac      = conc_i[j] / 100.0
                        depth     = frac * (1.0 - lib_raw[lb])
                        stack_bot = stack_top - depth
                        lbl_str   = (f'{lb}  {conc_i[j]:.1f} ± {err_i[j]:.1f}%'
                                     if err_i is not None else f'{lb}  {conc_i[j]:.1f}%')
                        col = self._an_colors[rank % len(self._an_colors)]
                        ax_top.fill_between(xaxis, stack_bot, stack_top,
                                            alpha=0.65, color=col, label=lbl_str)
                        ax_top.plot(xaxis, stack_bot, color=col, lw=0.5, alpha=0.9)
                        stack_top = stack_bot
                    if other_item is not None:
                        tot_frac, wsum = other_item
                        avg_raw   = wsum / tot_frac
                        depth     = tot_frac * (1.0 - avg_raw)
                        stack_bot = stack_top - depth
                        ax_top.fill_between(xaxis, stack_bot, stack_top, alpha=0.65,
                                            color=_AN_COLOR_OTHER,
                                            label=f'Other  {tot_frac*100:.1f}%')
                        ax_top.plot(xaxis, stack_bot, color=_AN_COLOR_OTHER,
                                    lw=0.5, alpha=0.9)
                        stack_top = stack_bot
                    _shade_excluded(ax_top)
                    y_floor = float(np.nanmin(stack_top))
                    ax_top.set_ylim(max(y_floor * 0.98, 0.0), 1.02)
                    ax_top.set_ylabel('Emissivity (stacked)')
                else:
                    n_shown = len(above) + (1 if other_item is not None else 0)
                    for rank, (j, lb) in enumerate(above):
                        col          = self._an_colors[rank % len(self._an_colors)]
                        frac         = conc_i[j] / 100.0
                        spec_display = (1.0 - frac) + lib_unit[lb] * frac + offset_val * rank
                        lbl_str      = (f'{lb}  {conc_i[j]:.1f} ± {err_i[j]:.1f}%'
                                        if err_i is not None else f'{lb}  {conc_i[j]:.1f}%')
                        ax_top.plot(xaxis, spec_display, color=col, lw=1.0, label=lbl_str)
                    if other_item is not None:
                        tot_frac, wsum = other_item
                        avg_raw   = wsum / tot_frac
                        r_range   = avg_raw[wn_mask]
                        lo, hi    = float(r_range.min()), float(r_range.max())
                        avg_normd = (avg_raw - lo) / (hi - lo + 1e-12)
                        spec_disp = (1.0 - tot_frac) + avg_normd * tot_frac
                        spec_disp += offset_val * len(above)
                        ax_top.plot(xaxis, spec_disp, lw=1.0, ls='--',
                                    color=_AN_COLOR_OTHER,
                                    label=f'Other  {tot_frac*100:.1f}%')
                    _shade_excluded(ax_top)
                    y_top = 1.05 + offset_val * max(0, n_shown - 1)
                    ax_top.set_ylim(0.0, y_top)
                    ax_top.set_ylabel('Scaled Emissivity')

            ax_top.axhline(1.0, color='k', lw=0.8, ls='--', zorder=0)
            ax_top.set_title(title)
            ax_top.tick_params(labelbottom=False)
            ax_top.legend(fontsize=8)
            _add_top_axis(ax_top)

        # Measured / modeled panel
        ax_bot.axhline(1.0, color='k', lw=0.8, ls='--', zorder=0)
        if meas_i is not None:
            ax_bot.plot(xaxis, meas_i, 'k', lw=1.0, label='Measured')
            ydata = meas_i[wn_mask] if wn_mask.any() else meas_i
            ymin  = float(np.nanmin(ydata)) * 0.95
            ymax  = float(np.nanmax(ydata)) * 1.05
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                ax_bot.set_ylim(max(ymin, 0.0), min(ymax, 1.1))
        if model_i is not None:
            ax_bot.plot(xaxis, model_i, c='m', lw=1.0, ls='-', label='Modeled')
        _shade_excluded(ax_bot)
        ax_bot.set_ylabel('Emissivity')
        ax_bot.legend(fontsize=8)
        ax_bot.set_xlim(xaxis.min(), xaxis.max())
        ax_bot.invert_xaxis()
        if ax_top is None:
            ax_bot.set_title(title)
        if ax_res is None:
            ax_bot.set_xlabel('Wavenumber [cm⁻¹]')
        else:
            ax_bot.tick_params(labelbottom=False)
        if ax_top is None and ax_res is None:
            _add_top_axis(ax_bot)

        # Residual panel
        if ax_res is not None and meas_i is not None and model_i is not None:
            resid_i = meas_i - model_i
            ax_res.plot(xaxis, resid_i, color='k', lw=1.0)
            ax_res.axhline(0, color='k', lw=0.8, ls='--')
            _shade_excluded(ax_res)
            ax_res.set_ylabel('Residual')
            ax_res.set_xlabel('Wavenumber [cm⁻¹]')

        self._an_update_limit_states()
        self._an_apply_limits()
        self._an_canvas.draw_idle()
        self._an_refresh_pie()

    def _an_open_limits_dialog(self) -> None:
        if (self._an_limits_dialog is not None
                and self._an_limits_dialog.winfo_exists()):
            self._an_limits_dialog.lift()
            self._an_limits_dialog.focus_force()
            return
        self._an_limits_dialog = AxisLimitsDialog(self)

    def _an_on_auto_toggle(self, which: str) -> None:
        """Read back current axis limits into vars when auto is disabled; then redraw."""
        if which == 'x' and not self._an_xlim_auto.get() and self._an_ax_main is not None:
            lo, hi = self._an_ax_main.get_xlim()
            self._an_xlim_lo_var.set(f'{lo:.1f}')
            self._an_xlim_hi_var.set(f'{hi:.1f}')
        elif which == 'y_top' and not self._an_ylim_top_auto.get() and self._an_ax_top is not None:
            lo, hi = self._an_ax_top.get_ylim()
            self._an_ylim_top_lo_var.set(f'{lo:.3f}')
            self._an_ylim_top_hi_var.set(f'{hi:.3f}')
        elif which == 'y_main' and not self._an_ylim_main_auto.get() and self._an_ax_main is not None:
            lo, hi = self._an_ax_main.get_ylim()
            self._an_ylim_main_lo_var.set(f'{lo:.3f}')
            self._an_ylim_main_hi_var.set(f'{hi:.3f}')
        elif which == 'y_resid' and not self._an_ylim_resid_auto.get() and self._an_ax_resid is not None:
            lo, hi = self._an_ax_resid.get_ylim()
            self._an_ylim_resid_lo_var.set(f'{lo:.4f}')
            self._an_ylim_resid_hi_var.set(f'{hi:.4f}')
        self._an_apply_and_redraw()

    def _an_update_limit_states(self) -> None:
        """Notify the limits dialog (if open) to refresh its widget enable/disable state."""
        if (self._an_limits_dialog is not None
                and self._an_limits_dialog.winfo_exists()):
            self._an_limits_dialog.update_states()

    def _an_apply_limits(self) -> None:
        """Apply stored manual axis limits to the current plot axes (no redraw)."""
        if not self._an_xlim_auto.get() and self._an_ax_main is not None:
            try:
                lo, hi = float(self._an_xlim_lo_var.get()), float(self._an_xlim_hi_var.get())
                if lo < hi:
                    for ax in self._an_fig.axes:
                        ax.set_xlim(lo, hi)
            except ValueError:
                pass

        if not self._an_ylim_top_auto.get() and self._an_ax_top is not None:
            try:
                lo, hi = float(self._an_ylim_top_lo_var.get()), float(self._an_ylim_top_hi_var.get())
                if lo < hi:
                    self._an_ax_top.set_ylim(lo, hi)
            except ValueError:
                pass

        if not self._an_ylim_main_auto.get() and self._an_ax_main is not None:
            try:
                lo, hi = float(self._an_ylim_main_lo_var.get()), float(self._an_ylim_main_hi_var.get())
                if lo < hi:
                    self._an_ax_main.set_ylim(lo, hi)
            except ValueError:
                pass

        if not self._an_ylim_resid_auto.get() and self._an_ax_resid is not None:
            try:
                lo, hi = float(self._an_ylim_resid_lo_var.get()), float(self._an_ylim_resid_hi_var.get())
                if lo < hi:
                    self._an_ax_resid.set_ylim(lo, hi)
            except ValueError:
                pass

    def _an_apply_and_redraw(self) -> None:
        self._an_apply_limits()
        self._an_canvas.draw_idle()

    def _an_refresh_table(self) -> None:
        r    = self._sma_result
        tree = self._an_tree

        for item in tree.get_children():
            tree.delete(item)
        tree['columns'] = ()

        if r is None or self._sma_label is None:
            return

        lbl           = self._sma_label
        sample_labels = r.get('sample_labels', [])
        if lbl not in sample_labels:
            return
        idx        = sample_labels.index(lbl)
        endmembers = r.get('labels', [])
        normconc   = r.get('normconc')

        if not endmembers or normconc is None:
            return

        cols = ['Endmember', 'Norm. conc. (%)']
        tree['columns'] = cols
        tree['show']    = 'headings'
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=160)

        for em_name, val in sorted(zip(endmembers, normconc[idx]),
                                   key=lambda x: x[1], reverse=True):
            tree.insert('', tk.END, values=(em_name, f'{val:.2f}'))

        rms = r.get('rms')
        if rms is not None and idx < len(rms):
            tree.insert('', tk.END, values=('RMS', f'{rms[idx]:.5f}'))

    def _an_refresh_pie(self) -> None:
        fig    = self._an_pie_fig
        canvas = self._an_pie_canvas
        fig.clear()

        r = self._sma_result
        if r is None or self._sma_label is None:
            canvas.draw_idle()
            return
        lbl           = self._sma_label
        sample_labels = r.get('sample_labels', [])
        if lbl not in sample_labels:
            canvas.draw_idle()
            return
        idx = sample_labels.index(lbl)

        use_group = self._an_group_var.get()
        use_error = self._an_error_var.get()
        try:
            threshold = float(self._an_threshold_var.get())
        except (tk.TclError, ValueError):
            threshold = 5.0

        gp = r.get('grouped') if use_group and r.get('grouped') else None
        if gp is not None:
            disp_normconc  = np.atleast_2d(gp['grouped_normconc'])
            disp_labels    = list(gp['grouped_labels'])
            disp_normerror = (np.atleast_2d(gp['grouped_normerror'])
                              if 'grouped_normerror' in gp else None)
        else:
            nc = r.get('normconc')
            if nc is None:
                canvas.draw_idle()
                return
            disp_normconc  = np.atleast_2d(nc)
            disp_labels    = r.get('labels', [])
            disp_normerror = (np.atleast_2d(r['normerror'])
                              if 'normerror' in r else None)

        if not disp_labels:
            canvas.draw_idle()
            return

        conc_i = disp_normconc[idx]
        err_i  = disp_normerror[idx] if (use_error and disp_normerror is not None) else None
        order  = np.argsort(conc_i)[::-1]

        pie_labels, pie_values, pie_colors = [], [], []
        color_idx = 0
        for j in order:
            if conc_i[j] >= threshold:
                lbl_str = (f'{disp_labels[j]}\n{conc_i[j]:.1f} ± {err_i[j]:.1f}%'
                           if err_i is not None
                           else f'{disp_labels[j]}\n{conc_i[j]:.1f}%')
                pie_labels.append(lbl_str)
                pie_values.append(float(conc_i[j]))
                pie_colors.append(self._an_colors[color_idx % len(self._an_colors)])
                color_idx += 1

        remainder = 100.0 - sum(pie_values)
        if remainder > 0.5:
            pie_labels.append(f'Other\n{remainder:.1f}%')
            pie_values.append(remainder)
            pie_colors.append(_AN_COLOR_OTHER)

        if not pie_values:
            canvas.draw_idle()
            return

        bb_normconc    = r.get('bb_normconc')
        slope_normconc = r.get('slope_normconc')
        bb_pct    = float(np.asarray(bb_normconc)[idx]) if bb_normconc is not None else 0.0
        sl_pct    = float(np.asarray(slope_normconc)[idx]) if slope_normconc is not None else 0.0
        delta_t   = r.get('delta_t_estimated')
        dt_pct    = float(np.asarray(delta_t)[idx]) if delta_t is not None else float('nan')
        mode_str  = '  [grouped]' if gp is not None else ''
        if slope_normconc is not None:
            _dt_str = f'  ΔT≈{dt_pct:.1f} K' if np.isfinite(dt_pct) else ''
            sl_str  = f'  |  Slope: {sl_pct:.1f}%{_dt_str}'
        else:
            sl_str  = ''

        fig.suptitle(
            f'{lbl}\nendmember composition{mode_str}\nBB: {bb_pct:.1f}%{sl_str}',
            fontsize=8,
        )
        ax = fig.add_subplot(111)
        ax.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
            textprops={'fontsize': 8},
        )
        ax.axis('equal')
        ax.set_position([0.15, 0.08, 0.7, 0.78])
        canvas.draw_idle()

    def _sma_plots_dir(self) -> Path:
        if self._fdir:
            base = Path(self._fdir)
        elif self._fdirs:
            base = Path(self._fdirs[0]).parent
        else:
            base = Path('.')
        out = base / 'sma_plots'
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _safe_label(lbl: str) -> str:
        return lbl.replace(' ', '_').replace('/', '-').replace('\\', '-')

    def _save_sma_sample(self, lbl: str, out_dir: Path) -> None:
        safe = self._safe_label(lbl)
        self._an_fig.savefig(str(out_dir / f'sma_{safe}.png'),
                             dpi=150, bbox_inches='tight')
        self._an_pie_fig.savefig(str(out_dir / f'sma_{safe}_pie.png'),
                                 dpi=150, bbox_inches='tight')

    def _an_save_current(self) -> None:
        if self._sma_label is None:
            return
        out_dir = self._sma_plots_dir()
        self._save_sma_sample(self._sma_label, out_dir)
        logging.info("Saved SMA plots for '%s' to %s", self._sma_label, out_dir)

    def _an_save_all(self) -> None:
        r = self._sma_result
        if r is None:
            return
        labels  = r.get('sample_labels', [])
        out_dir = self._sma_plots_dir()
        orig    = self._sma_label
        for lbl in labels:
            self._sma_label = lbl
            self._an_refresh_plot()
            self._save_sma_sample(lbl, out_dir)
        self._sma_label = orig
        self._an_refresh_plot()
        logging.info("Saved %d SMA plot pairs to %s", len(labels), out_dir)


# ---------------------------------------------------------------------------

def launch() -> None:
    app = EmissionLWIR()
    app.mainloop()


if __name__ == '__main__':
    launch()
