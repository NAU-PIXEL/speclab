#!/usr/bin/env python3
"""
ReflectanceVSWIR — GUI for VSWIR reflectance data processing.

Layout
------
Top     : Load Data | Load Library   (single narrow row, spans all columns) 2
Main    : Data panel (left) | spectral plot (centre, tall) | Library panel (right)

Data panel (left)
-----------------
Buttons above  : Add | Add as Group
Listbox        : loaded spectra

Library panel (right)
---------------------
Filters above  : 3 × (Filter by: <keyword dropdown>   <value dropdown>)
Listbox        : library spectra matching active filters
Buttons below  : Add | Add as Group
"""

import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import cmcrameri
import yaml

# Allow running directly as a script in addition to normal entry points.
if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'speclab'

from . import __version__
from .functions import remove_continuum, band_parameters, smooth_spectrum, detect_bands
from .utils import (
    loadASD, loadReflectanceCSV, saveReflectanceCSV, readDVhdf, _WL_COLUMN_NAMES,
)
from .SpeclibViewer import SpeclibViewer

plt.rcParams.update({
    'font.size':             11,
    'lines.linewidth':       1.0,
    'axes.prop_cycle':       plt.cycler(color=plt.cm.Dark2.colors),
    'xtick.direction':       'in',
    'xtick.top':             True,
    'xtick.labelsize':       11,
    'ytick.direction':       'in',
    'ytick.right':           True,
    'ytick.labelsize':       11,
    'axes.grid':             False,
    'axes.axisbelow':        True,
    'axes.labelsize':        12,
    'axes.titlesize':        12,
    'grid.linestyle':        '--',
    'axes.formatter.limits': (-4, 4),
    'errorbar.capsize':      2,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL = 'All'

# Metadata fields that carry spectral data or internal bookkeeping — excluded
# from the filter keyword discovery scan.
_LIB_NON_FILTER_FIELDS: frozenset[str] = frozenset({
    'data', 'xaxis', 'name', 'label', 'sample_name', 'source',
})

# Pretty-print overrides for known field names (field_name → display label).
_LIB_FIELD_DISPLAY: dict[str, str] = {
    # USGS splib07 fields
    'chapter':              'Chapter',
    'mineral_name':         'Mineral',
    'mineral_type':         'Mineral Type',
    'mineral':              'Mineral (full)',
    'meas_type':            'Meas. Type',
    'sample_id':            'Sample ID',
    'spectrometer':         'Spectrometer',
    'original_donor':       'Donor',
    'source_library':       'Library',
    'collection_locality':  'Locality',
    # CRISM spectral library fields
    'type':                 'Instrument',
    'body':                 'Body',
    'material':             'Material',
    'mineral_family':       'Min. Family',
    'reference':            'Reference',
    'current_location':     'Repository',
    'collection_location':  'Origin',
}

_N_FILTER_ROWS = 3

# Display modes available in the "Add as Group" dropdown.
_GROUP_MODE_VALUES: list[str] = ['Median ± std', 'All spectra']

# ---------------------------------------------------------------------------
# Color scheme system
# ---------------------------------------------------------------------------

def _lighten_colors(colors: list, factor: float = 0.5) -> list:
    out = []
    for c in colors:
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        out.append((r + factor * (1.0 - r), g + factor * (1.0 - g), b + factor * (1.0 - b)))
    return out


def _cmc_colors(name: str) -> list | None:
    """Return the discrete color list for a cmcrameri S colormap, or None if absent."""
    try:
        return list(getattr(cmcrameri.cm, name).colors)
    except AttributeError:
        return None


_tab20_dark  = [plt.cm.tab20(i) for i in range(0,  20, 2)]
_tab20_light = [plt.cm.tab20(i) for i in range(1,  20, 2)]
_dark2       = list(plt.cm.Dark2.colors)

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


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _df_to_xaxis_spectra(
    df: pd.DataFrame,
    path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract ``(xaxis, spectra_dict)`` from a wide-format reflectance DataFrame."""
    wl_col = next(c for c in df.columns if c.strip().lower() in _WL_COLUMN_NAMES)
    xaxis  = df[wl_col].to_numpy(dtype=np.float64)
    spectra: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col == wl_col:
            continue
        try:
            spectra[col] = df[col].to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Column '{col}' in '{path.name}' contains non-numeric data."
            ) from exc
    return xaxis, spectra


def _read_vswir_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a generic VSWIR reflectance CSV → ``(xaxis, spectra_dict)``."""
    return _df_to_xaxis_spectra(loadReflectanceCSV(path), path)


def _read_vswir_asd(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read an ASD tab-separated text export → ``(xaxis, spectra_dict)``."""
    return _df_to_xaxis_spectra(loadASD(path), path)


# ---------------------------------------------------------------------------
# Modal dialogs
# ---------------------------------------------------------------------------

class ContinuumRemovalDialog(tk.Toplevel):
    """
    Modal dialog for configuring convex-hull continuum removal.

    After ``wait_window`` returns, check ``self.cancelled``.  If ``False``,
    ``self.wl_range`` holds the chosen wavelength bounds (or ``None`` for the
    full spectrum).

    Attributes
    ----------
    cancelled : bool
        ``True`` if the user dismissed without running.
    wl_range : tuple[float, float] or None
        Wavelength range (nm) passed to ``remove_continuum``.
        ``None`` means no restriction — full spectrum.
    """

    def __init__(
        self,
        master: tk.Misc,
        default_range: tuple[float, float] = (400.0, 2500.0),
        last_range:    tuple[float, float] | None = None,
    ) -> None:
        super().__init__(master)
        self.title('Remove Continuum')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)

        self.cancelled: bool                      = True
        self.wl_range:  tuple[float, float] | None = None

        lo0, hi0 = last_range if last_range is not None else default_range
        self._lo_var   = tk.DoubleVar(value=lo0)
        self._hi_var   = tk.DoubleVar(value=hi0)
        self._full_var = tk.BooleanVar(value=(last_range is None))

        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)

        # Wavelength range row
        rng_row = ttk.Frame(frm)
        rng_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(rng_row, text='Wavelength range:').pack(side=tk.LEFT, padx=(0, 8))
        self._lo_sb = ttk.Spinbox(
            rng_row, textvariable=self._lo_var,
            from_=100.0, to=3000.0, increment=10.0, width=9, format='%.1f',
        )
        self._lo_sb.pack(side=tk.LEFT)
        ttk.Label(rng_row, text='–').pack(side=tk.LEFT, padx=6)
        self._hi_sb = ttk.Spinbox(
            rng_row, textvariable=self._hi_var,
            from_=100.0, to=3000.0, increment=10.0, width=9, format='%.1f',
        )
        self._hi_sb.pack(side=tk.LEFT)
        ttk.Label(rng_row, text='nm').pack(side=tk.LEFT, padx=(6, 0))

        # Full-spectrum toggle
        ttk.Checkbutton(
            frm,
            text='Full spectrum (no restriction)',
            variable=self._full_var,
            command=self._on_full_toggle,
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        btn_row = ttk.Frame(frm)
        btn_row.pack()
        ttk.Button(btn_row, text='Run',    command=self._run).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT)

        self._on_full_toggle()

    def _on_full_toggle(self) -> None:
        state = 'disabled' if self._full_var.get() else 'normal'
        self._lo_sb.config(state=state)
        self._hi_sb.config(state=state)

    def _run(self) -> None:
        if self._full_var.get():
            self.wl_range = None
        else:
            lo, hi = self._lo_var.get(), self._hi_var.get()
            if lo >= hi:
                messagebox.showerror(
                    'Invalid range',
                    'Lower wavelength must be less than upper wavelength.',
                    parent=self,
                )
                return
            self.wl_range = (lo, hi)
        self.cancelled = False
        self.destroy()


class BandParametersDialog(tk.Toplevel):
    """
    Modal dialog for selecting preset features and adjusting their shoulder
    windows before computing band parameters.

    Features are pre-checked when they are already active as plot markers.
    Custom markers (including unmatched identified bands) are shown under a
    separate group and are pre-checked by default.
    Each row exposes the shoulder wavelength range used for continuum removal
    and parameter extraction.

    Scope radio buttons let the user choose whether to analyse only plotted
    spectra or all loaded/available spectra on each side.

    Attributes
    ----------
    cancelled : bool
        ``True`` if the user dismissed without running.
    selected : list[dict]
        Dicts with keys ``'name'``, ``'group'``, ``'wl_range'`` for every
        checked feature, in YAML order.
    data_scope : str
        ``'plotted'`` or ``'all'`` — which data spectra to analyse.
    lib_scope : str
        ``'plotted'`` or ``'all'`` — which library spectra to analyse.
    """

    def __init__(
        self,
        master:            tk.Misc,
        preset_data:       list[dict],
        active_names:      set[str],
        *,
        n_data_plotted:    int = 0,
        n_data_available:  int = 0,
        n_lib_plotted:     int = 0,
        n_lib_available:   int = 0,
    ) -> None:
        super().__init__(master)
        self.title('Band Parameters')
        self.resizable(False, True)
        self.grab_set()
        self.transient(master)

        self.cancelled:  bool       = True
        self.selected:   list[dict] = []
        self.data_scope: str        = 'plotted'
        self.lib_scope:  str        = 'plotted'

        self._preset_data      = preset_data
        self._active_names     = active_names
        self._rows:  list[dict] = []   # per-row vars

        self._n_data_plotted   = n_data_plotted
        self._n_data_available = n_data_available
        self._n_lib_plotted    = n_lib_plotted
        self._n_lib_available  = n_lib_available

        self._data_scope_var = tk.StringVar(value='plotted')
        self._lib_scope_var  = tk.StringVar(value='plotted')

        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=(12, 10))
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Column header (fixed, above scrollable area) ──────────────────
        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hdr, text='Feature',      width=30).pack(side=tk.LEFT)
        ttk.Label(hdr, text='Window lo (nm)', width=13, anchor='center').pack(side=tk.LEFT)
        ttk.Label(hdr, text='Window hi (nm)', width=13, anchor='center').pack(side=tk.LEFT)
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 4))

        # ── Scrollable feature rows ───────────────────────────────────────
        scroll_outer = ttk.Frame(outer)
        scroll_outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(scroll_outer, borderwidth=0, highlightthickness=0,
                           height=min(400, len(self._preset_data) * 28 + 60))
        vsb = ttk.Scrollbar(scroll_outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rows_frame = ttk.Frame(canvas)
        _cwin = canvas.create_window((0, 0), window=rows_frame, anchor='nw')
        rows_frame.bind(
            '<Configure>',
            lambda _e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(_cwin, width=e.width))

        current_group: str | None = None
        for preset in self._preset_data:
            name  = preset['name']
            group = preset.get('group', '')
            if group != current_group:
                current_group = group
                g_row = ttk.Frame(rows_frame)
                g_row.pack(fill=tk.X, pady=(6, 0))
                ttk.Label(g_row, text=group,
                          font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT)
                ttk.Separator(rows_frame, orient=tk.HORIZONTAL).pack(
                    fill=tk.X, pady=(1, 2))

            # Default wl_range from YAML; fall back to centre ± 3*fwhm
            if 'wl_range' in preset:
                lo0, hi0 = float(preset['wl_range'][0]), float(preset['wl_range'][1])
            else:
                lo0 = float(preset['wavelength']) - 3 * float(preset['fwhm'])
                hi0 = float(preset['wavelength']) + 3 * float(preset['fwhm'])

            enabled_var = tk.BooleanVar(value=(name in self._active_names))
            lo_var      = tk.DoubleVar(value=lo0)
            hi_var      = tk.DoubleVar(value=hi0)
            self._rows.append({'name': name, 'group': group,
                                'enabled': enabled_var,
                                'lo_var': lo_var, 'hi_var': hi_var})

            row = ttk.Frame(rows_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(row, text=name, variable=enabled_var,
                            width=30).pack(side=tk.LEFT)
            for var in (lo_var, hi_var):
                ttk.Spinbox(row, textvariable=var, from_=100.0, to=3000.0,
                            increment=5.0, width=9, format='%.1f',
                            ).pack(side=tk.LEFT, padx=(4, 0))

        # ── Scope ─────────────────────────────────────────────────────────
        scope_lf = ttk.LabelFrame(outer, text='Scope', padding=(6, 4))
        scope_lf.pack(fill=tk.X, pady=(8, 0))

        data_row = ttk.Frame(scope_lf)
        data_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(data_row, text='Data:', width=8, anchor='w').pack(side=tk.LEFT)
        ttk.Radiobutton(
            data_row, text=f'Plotted ({self._n_data_plotted})',
            variable=self._data_scope_var, value='plotted',
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            data_row, text=f'All loaded ({self._n_data_available})',
            variable=self._data_scope_var, value='all',
        ).pack(side=tk.LEFT)

        lib_row = ttk.Frame(scope_lf)
        lib_row.pack(fill=tk.X)
        ttk.Label(lib_row, text='Library:', width=8, anchor='w').pack(side=tk.LEFT)
        ttk.Radiobutton(
            lib_row, text=f'Plotted ({self._n_lib_plotted})',
            variable=self._lib_scope_var, value='plotted',
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            lib_row, text=f'All available ({self._n_lib_available})',
            variable=self._lib_scope_var, value='all',
        ).pack(side=tk.LEFT)

        # ── Buttons ───────────────────────────────────────────────────────
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 6))
        btn_row = ttk.Frame(outer)
        btn_row.pack()
        ttk.Button(btn_row, text='Run',    command=self._run).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT)

    def _run(self) -> None:
        selected = []
        for row in self._rows:
            if not row['enabled'].get():
                continue
            lo, hi = row['lo_var'].get(), row['hi_var'].get()
            if lo >= hi:
                messagebox.showerror(
                    'Invalid range',
                    f"Feature '{row['name']}': lower wavelength must be less "
                    f"than upper wavelength.",
                    parent=self,
                )
                return
            selected.append({'name': row['name'], 'group': row['group'],
                             'wl_range': (lo, hi)})
        if not selected:
            messagebox.showwarning('No features selected',
                                   'Select at least one feature to compute.',
                                   parent=self)
            return
        self.selected   = selected
        self.data_scope = self._data_scope_var.get()
        self.lib_scope  = self._lib_scope_var.get()
        self.cancelled  = False
        self.destroy()


class SmoothingDialog(tk.Toplevel):
    """
    Modal dialog for applying spectral smoothing to all currently plotted spectra.

    After ``wait_window`` returns, check ``self.cancelled``.  If ``False``,
    ``self.params`` holds the smoothing parameters ready to unpack into
    :func:`smooth_spectrum`.

    Attributes
    ----------
    cancelled : bool
    params : dict
        Keys: ``smooth_method``, ``smooth_window_nm``, ``smooth_polyorder``.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title('Smooth Spectra')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)

        self.cancelled: bool = True
        self.params:    dict = {}

        self._method_var    = tk.StringVar(value='Savitzky-Golay')
        self._window_var    = tk.DoubleVar(value=30.0)
        self._polyorder_var = tk.IntVar(value=3)

        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=(12, 10))
        frm.pack(fill=tk.BOTH, expand=True)

        row1 = ttk.Frame(frm)
        row1.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(row1, text='Method:', width=10, anchor='w').pack(side=tk.LEFT)
        self._method_cb = ttk.Combobox(
            row1, textvariable=self._method_var,
            values=_SMOOTH_LABELS, state='readonly', width=18,
        )
        self._method_cb.pack(side=tk.LEFT, padx=(0, 12))
        self._method_cb.bind('<<ComboboxSelected>>', self._on_method_change)
        ttk.Label(row1, text='Window (nm):', anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(
            row1, textvariable=self._window_var,
            from_=1.0, to=500.0, increment=5.0,
            width=8, format='%.1f',
        ).pack(side=tk.LEFT, padx=(4, 0))

        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(row2, text='', width=10).pack(side=tk.LEFT)
        ttk.Label(row2, text='Poly order:', anchor='w').pack(side=tk.LEFT)
        self._polyorder_sb = ttk.Spinbox(
            row2, textvariable=self._polyorder_var,
            from_=1, to=9, increment=1, width=5,
        )
        self._polyorder_sb.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(
            row2,
            text='(Savitzky-Golay only — window in nm rounded to nearest odd sample count)',
            font=('TkDefaultFont', 8), foreground='gray',
        ).pack(side=tk.LEFT)

        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))
        btn_row = ttk.Frame(frm)
        btn_row.pack()
        ttk.Button(btn_row, text='Apply',  command=self._apply ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT)

        self._on_method_change()

    def _on_method_change(self, _event=None) -> None:
        state = 'normal' if self._method_var.get() == 'Savitzky-Golay' else 'disabled'
        self._polyorder_sb.config(state=state)

    def _apply(self) -> None:
        self.params = {
            'smooth_method':    _SMOOTH_METHODS[self._method_var.get()],
            'smooth_window_nm': self._window_var.get(),
            'smooth_polyorder': self._polyorder_var.get(),
        }
        self.cancelled = False
        self.destroy()


# Mapping from UI display label to smooth_spectrum() method string.
_SMOOTH_METHODS: dict[str, str] = {
    'Moving Average':  'moving_avg',
    'Moving Median':   'moving_median',
    'Savitzky-Golay':  'savgol',
    'Gaussian':        'gaussian',
}
_SMOOTH_LABELS: list[str] = list(_SMOOTH_METHODS.keys())


class BandIdentificationDialog(tk.Toplevel):
    """
    Modal pre-run dialog for the Band Identification workflow.

    Collects spectrum selection, smoothing parameters, peak-detection
    thresholds, and feature-matching tolerance.

    After ``wait_window`` returns, check ``self.cancelled``.  If ``False``:

    Attributes
    ----------
    cancelled : bool
    selected_names : set[str]
        Names of the spectra the user chose to analyse.
    params : dict
        Algorithm parameters ready to unpack into :func:`detect_bands`.
    """

    def __init__(
        self,
        master:    tk.Misc,
        all_items: list[dict],
        *,
        n_data_plotted:   int = 0,
        n_data_available: int = 0,
        n_lib_plotted:    int = 0,
        n_lib_available:  int = 0,
    ) -> None:
        super().__init__(master)
        self.title('Identify Bands')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)

        self.cancelled:      bool     = True
        self.selected_names: set[str] = set()
        self.params:         dict     = {}
        self.data_scope:     str      = 'plotted'
        self.lib_scope:      str      = 'plotted'

        self._all_items = all_items
        self._sel_vars: list[tk.BooleanVar] = [
            tk.BooleanVar(value=True) for _ in all_items
        ]

        self._n_data_plotted   = n_data_plotted
        self._n_data_available = n_data_available
        self._n_lib_plotted    = n_lib_plotted
        self._n_lib_available  = n_lib_available
        self._data_scope_var   = tk.StringVar(value='plotted')
        self._lib_scope_var    = tk.StringVar(value='plotted')

        self._method_var     = tk.StringVar(value='Savitzky-Golay')
        self._window_var     = tk.DoubleVar(value=30.0)
        self._polyorder_var  = tk.IntVar(value=3)
        self._prominence_var = tk.DoubleVar(value=0.020)
        self._width_var      = tk.DoubleVar(value=15.0)
        self._depth_var      = tk.DoubleVar(value=0.010)
        self._tolerance_var  = tk.DoubleVar(value=30.0)

        self._build()
        self.wait_window(self)

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=(10, 8))
        frm.pack(fill=tk.BOTH, expand=True)

        # ── Spectrum selection ────────────────────────────────────────────────
        sel_lf = ttk.LabelFrame(frm, text='Spectra to analyse', padding=(6, 4))
        sel_lf.pack(fill=tk.X, pady=(0, 8))

        outer  = ttk.Frame(sel_lf)
        outer.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0,
                           height=min(120, len(self._all_items) * 22 + 8))
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas)
        cwin  = canvas.create_window((0, 0), window=inner, anchor='nw')
        inner.bind('<Configure>',
                   lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(cwin, width=e.width))
        for item, var in zip(self._all_items, self._sel_vars):
            ttk.Checkbutton(inner, text=item['name'], variable=var).pack(
                anchor='w', padx=4, pady=1)

        sel_btn_row = ttk.Frame(sel_lf)
        sel_btn_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(sel_btn_row, text='Select All',
                   command=lambda: [v.set(True)  for v in self._sel_vars]
                   ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_btn_row, text='Clear All',
                   command=lambda: [v.set(False) for v in self._sel_vars]
                   ).pack(side=tk.LEFT)

        # ── Smoothing ─────────────────────────────────────────────────────────
        sm_lf = ttk.LabelFrame(frm, text='Smoothing', padding=(6, 4))
        sm_lf.pack(fill=tk.X, pady=(0, 8))

        sm_row1 = ttk.Frame(sm_lf)
        sm_row1.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(sm_row1, text='Method:', width=10, anchor='w').pack(side=tk.LEFT)
        self._method_cb = ttk.Combobox(
            sm_row1, textvariable=self._method_var,
            values=_SMOOTH_LABELS, state='readonly', width=18,
        )
        self._method_cb.pack(side=tk.LEFT, padx=(0, 12))
        self._method_cb.bind('<<ComboboxSelected>>', self._on_method_change)
        ttk.Label(sm_row1, text='Window (nm):', anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(sm_row1, textvariable=self._window_var,
                    from_=1.0, to=500.0, increment=5.0,
                    width=8, format='%.1f').pack(side=tk.LEFT, padx=(4, 0))

        sm_row2 = ttk.Frame(sm_lf)
        sm_row2.pack(fill=tk.X)
        ttk.Label(sm_row2, text='', width=10).pack(side=tk.LEFT)   # spacer aligns with Method
        ttk.Label(sm_row2, text='Poly order:', anchor='w').pack(side=tk.LEFT)
        self._polyorder_sb = ttk.Spinbox(
            sm_row2, textvariable=self._polyorder_var,
            from_=1, to=9, increment=1, width=5,
        )
        self._polyorder_sb.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(
            sm_row2,
            text='(Savitzky-Golay only — window in nm rounded to nearest odd sample count)',
            font=('TkDefaultFont', 8), foreground='gray',
        ).pack(side=tk.LEFT)

        # ── Peak detection ────────────────────────────────────────────────────
        pk_lf = ttk.LabelFrame(frm, text='Peak detection', padding=(6, 4))
        pk_lf.pack(fill=tk.X, pady=(0, 8))

        pk_row = ttk.Frame(pk_lf)
        pk_row.pack(fill=tk.X)
        for label, var, lo, hi, inc, fmt in [
            ('Min prominence:',  self._prominence_var, 0.001, 1.0,   0.005, '%.3f'),
            ('Min width (nm):',  self._width_var,      1.0,   500.0, 5.0,   '%.1f'),
            ('Min band depth:',  self._depth_var,      0.001, 1.0,   0.005, '%.3f'),
        ]:
            ttk.Label(pk_row, text=label, anchor='w').pack(side=tk.LEFT, padx=(0, 2))
            ttk.Spinbox(pk_row, textvariable=var,
                        from_=lo, to=hi, increment=inc,
                        width=8, format=fmt).pack(side=tk.LEFT, padx=(0, 14))

        # ── Feature matching ──────────────────────────────────────────────────
        mt_lf = ttk.LabelFrame(frm, text='Feature matching', padding=(6, 4))
        mt_lf.pack(fill=tk.X, pady=(0, 10))

        mt_row = ttk.Frame(mt_lf)
        mt_row.pack(fill=tk.X)
        ttk.Label(mt_row, text='Tolerance (nm):', anchor='w').pack(side=tk.LEFT)
        ttk.Spinbox(mt_row, textvariable=self._tolerance_var,
                    from_=1.0, to=200.0, increment=5.0,
                    width=8, format='%.1f').pack(side=tk.LEFT, padx=(4, 0))

        # ── Scope ─────────────────────────────────────────────────────────────
        scope_lf = ttk.LabelFrame(frm, text='Scope', padding=(6, 4))
        scope_lf.pack(fill=tk.X, pady=(0, 10))

        data_row = ttk.Frame(scope_lf)
        data_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(data_row, text='Data:', width=8, anchor='w').pack(side=tk.LEFT)
        ttk.Radiobutton(
            data_row, text=f'Plotted ({self._n_data_plotted})',
            variable=self._data_scope_var, value='plotted',
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            data_row, text=f'All loaded ({self._n_data_available})',
            variable=self._data_scope_var, value='all',
        ).pack(side=tk.LEFT)

        lib_row = ttk.Frame(scope_lf)
        lib_row.pack(fill=tk.X)
        ttk.Label(lib_row, text='Library:', width=8, anchor='w').pack(side=tk.LEFT)
        ttk.Radiobutton(
            lib_row, text=f'Plotted ({self._n_lib_plotted})',
            variable=self._lib_scope_var, value='plotted',
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Radiobutton(
            lib_row, text=f'All available ({self._n_lib_available})',
            variable=self._lib_scope_var, value='all',
        ).pack(side=tk.LEFT)

        # ── Buttons ───────────────────────────────────────────────────────────
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 8))
        run_row = ttk.Frame(frm)
        run_row.pack()
        ttk.Button(run_row, text='Run',    command=self._run   ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(run_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT)

        self._on_method_change()

    def _on_method_change(self, _event=None) -> None:
        state = 'normal' if self._method_var.get() == 'Savitzky-Golay' else 'disabled'
        self._polyorder_sb.config(state=state)

    def _run(self) -> None:
        selected = {item['name'] for item, var in zip(self._all_items, self._sel_vars)
                    if var.get()}
        if not selected:
            messagebox.showwarning(
                'No spectra selected',
                'Select at least one spectrum to analyse.',
                parent=self,
            )
            return
        self.selected_names = selected
        self.params = {
            'smooth_method':      _SMOOTH_METHODS[self._method_var.get()],
            'smooth_window_nm':   self._window_var.get(),
            'smooth_polyorder':   self._polyorder_var.get(),
            'min_prominence':     self._prominence_var.get(),
            'min_width_nm':       self._width_var.get(),
            'min_depth':          self._depth_var.get(),
            'match_tolerance_nm': self._tolerance_var.get(),
        }
        self.data_scope = self._data_scope_var.get()
        self.lib_scope  = self._lib_scope_var.get()
        self.cancelled  = False
        self.destroy()


class BandIdentificationResultsDialog(tk.Toplevel):
    """
    Non-modal window listing auto-detected band candidates with checkboxes.

    The user reviews the merged, cross-spectrum candidate list and clicks
    "Add to Markers" to enable the selected presets or append custom markers.
    """

    def __init__(
        self,
        master:     tk.Misc,
        candidates: list[dict],
        on_add,                  # callable(list[dict]) -> None
    ) -> None:
        super().__init__(master)
        self.title('Band Identification — Results')
        self.resizable(True, True)

        self._candidates = candidates
        self._on_add     = on_add
        self._row_vars: list[tk.BooleanVar] = [
            tk.BooleanVar(value=True) for _ in candidates
        ]
        self._build()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=(8, 6))
        frm.pack(fill=tk.BOTH, expand=True)

        # ── Summary ───────────────────────────────────────────────────────────
        n_total   = len(self._candidates)
        n_matched = sum(1 for c in self._candidates if c.get('matched_name'))
        ttk.Label(
            frm,
            text=(f'{n_total} candidate{"s" if n_total != 1 else ""} — '
                  f'{n_matched} matched to presets, '
                  f'{n_total - n_matched} unmatched'),
        ).pack(anchor='w', pady=(0, 6))

        # ── Column headers ────────────────────────────────────────────────────
        hdr = ttk.Frame(frm)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text='Matched feature', width=24, anchor='w' ).pack(side=tk.LEFT)
        ttk.Label(hdr, text='Band min (nm)',   width=12, anchor='center').pack(side=tk.LEFT)
        ttk.Label(hdr, text='Depth',           width=8,  anchor='center').pack(side=tk.LEFT)
        ttk.Label(hdr, text='FWHM (nm)',       width=10, anchor='center').pack(side=tk.LEFT)
        ttk.Label(hdr, text='Seen in',                   anchor='w'     ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(2, 0))

        # ── Scrollable rows ───────────────────────────────────────────────────
        outer = ttk.Frame(frm)
        outer.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0,
                           height=min(340, len(self._candidates) * 26 + 10))
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rows_frame = ttk.Frame(canvas)
        cwin = canvas.create_window((0, 0), window=rows_frame, anchor='nw')
        rows_frame.bind(
            '<Configure>',
            lambda _e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        canvas.bind(
            '<Configure>',
            lambda e: canvas.itemconfigure(cwin, width=e.width),
        )

        for cand, var in zip(self._candidates, self._row_vars):
            name  = cand.get('matched_name') or '—'
            wl    = f"{cand['wl_min']:.1f}"
            depth = f"{cand['band_depth']:.4f}"
            fwhm  = f"{cand['fwhm']:.1f}"
            seen_list = cand.get('seen_in', [])
            if len(seen_list) <= 2:
                seen = ', '.join(seen_list)
            else:
                seen = f'{len(seen_list)} spectra'

            row = ttk.Frame(rows_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(row, text=name, variable=var, width=24).pack(side=tk.LEFT)
            ttk.Label(row, text=wl,    width=12, anchor='center').pack(side=tk.LEFT)
            ttk.Label(row, text=depth, width=8,  anchor='center').pack(side=tk.LEFT)
            ttk.Label(row, text=fwhm,  width=10, anchor='center').pack(side=tk.LEFT)
            ttk.Label(row, text=seen,            anchor='w'     ).pack(side=tk.LEFT, padx=(4, 0))

        # ── Select all / clear all ────────────────────────────────────────────
        sel_row = ttk.Frame(frm)
        sel_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(sel_row, text='Select All',
                   command=lambda: [v.set(True)  for v in self._row_vars]
                   ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_row, text='Clear All',
                   command=lambda: [v.set(False) for v in self._row_vars]
                   ).pack(side=tk.LEFT)

        # ── Action buttons ────────────────────────────────────────────────────
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(6, 6))
        act_row = ttk.Frame(frm)
        act_row.pack()
        ttk.Button(act_row, text='Add to Markers',
                   command=self._on_add_clicked).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(act_row, text='Close',
                   command=self.destroy).pack(side=tk.LEFT)

    def _on_add_clicked(self) -> None:
        selected = [c for c, v in zip(self._candidates, self._row_vars) if v.get()]
        if selected:
            self._on_add(selected)
        self.destroy()


# Ordered spec for every band parameter column.
# (result_key, column_label_template, unit_scaled, format_spec)
# unit_scaled=True  → value is in nm internally; multiply by 1e-3 when unit='µm'
# unit_scaled=False → dimensionless; no scaling
_BAND_METRIC_SPEC: list[tuple[str, str, bool, str]] = [
    ('wl_center',          'Band center\n({unit})', True,  '.2f'),
    ('wl_min',             'Band min\n({unit})',    True,  '.2f'),
    ('band_depth',         'Depth',                 False, '.4f'),
    ('fwhm',               'FWHM\n({unit})',        True,  '.2f'),
    ('base_width',         'Base width\n({unit})',  True,  '.2f'),
    ('band_area',          'Band area\n({unit})',   True,  '.2f'),
    ('band_area_ratio',    'Area ratio',            False, '.4f'),
    ('asymmetry_hw',       'Asym HW',              False, '.4f'),
    ('asymmetry_centroid', 'Asym centroid',         False, '.4f'),
]


# Clean single-line display labels for metrics (used in dropdowns and axis labels).
# Keys match _BAND_METRIC_SPEC; '{unit}' is replaced at runtime.
_METRIC_CLEAN_LABELS: dict[str, str] = {
    'wl_center':          'Band center ({unit})',
    'wl_min':             'Band min ({unit})',
    'band_depth':         'Depth',
    'fwhm':               'FWHM ({unit})',
    'base_width':         'Base width ({unit})',
    'band_area':          'Band area ({unit})',
    'band_area_ratio':    'Area ratio',
    'asymmetry_hw':       'Asymmetry (HW)',
    'asymmetry_centroid': 'Asymmetry (centroid)',
}


class BandVizWindow(tk.Toplevel):
    """
    Non-modal visualization window for band parameter results.

    Two tabs:

    **Scatter** — two modes selectable via radio buttons:

    * *Metric vs Metric* — each point is a (spectrum × band) pair, X and Y
      are independently chosen metrics.  Points are coloured by band name,
      feature group, or left uniform.
    * *Band vs Band* — each point is one spectrum; X is one band's value for
      the chosen metric, Y is another band's value.  Useful for correlation
      between two specific absorption features across samples.

    Clicking a point (while no pan/zoom tool is active) toggles a persistent
    annotation.  "Clear labels" removes all annotations.

    **Violin** — one violin per selected band showing the distribution of a
    chosen metric across all spectra; individual data points overlaid as a
    jittered strip.

    The window auto-plots the scatter on open.

    Parameters
    ----------
    master : tk.Misc
        Parent widget.
    features : list[dict]
        Feature definitions (keys: ``'name'``, ``'group'``, ``'wl_range'``).
    results : dict[str, dict]
        Band parameter results: ``{spectrum_name: {feat_name: dict | None}}``.
    unit : str
        Wavelength unit for axis labels (``'nm'`` or ``'µm'``).
    """

    def __init__(
        self,
        master:   tk.Misc,
        features: list[dict],
        results:  dict[str, dict],
        unit:     str = 'nm',
        *,
        sources:  dict[str, str] | None = None,
    ) -> None:
        super().__init__(master)
        self.title('Band Parameter Visualization')
        self.geometry('860x860')
        self.resizable(True, True)

        self._features = features
        self._results  = results
        self._unit     = unit
        self._scale    = 1e-3 if unit == 'µm' else 1.0
        self._sources  = sources or {}

        self._metric_labels: dict[str, str] = {
            k: tmpl.replace('{unit}', unit)
            for k, tmpl in _METRIC_CLEAN_LABELS.items()
        }
        self._label_to_key: dict[str, str] = {v: k for k, v in self._metric_labels.items()}

        self._records: list[dict] = self._build_records()

        self._scatter_xdata:        np.ndarray | None = None
        self._scatter_ydata:        np.ndarray | None = None
        self._scatter_names:        list[str]  | None = None
        self._scatter_annotations:  dict               = {}
        self._scatter_legend_handles: list             = []
        self._legend_win:           tk.Toplevel | None = None

        self._build()
        self.after(60, self._plot_scatter)   # auto-plot on open

    # ── Data ──────────────────────────────────────────────────────────────────

    def _build_records(self) -> list[dict]:
        """Flatten results into per-(spectrum × band) dicts with pre-scaled values."""
        records = []
        for sp_name, feat_results in self._results.items():
            for feat in self._features:
                bp = feat_results.get(feat['name'])
                if bp is None:
                    continue
                rec: dict = {
                    'spectrum': sp_name,
                    'feature':  feat['name'],
                    'group':    feat.get('group', ''),
                    'source':   self._sources.get(sp_name, ''),
                }
                for key, _, scaled, _ in _BAND_METRIC_SPEC:
                    raw = bp.get(key, np.nan)
                    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                        rec[key] = np.nan
                    else:
                        rec[key] = float(raw) * (self._scale if scaled else 1.0)
                records.append(rec)
        return records

    # ── UI construction ───────────────────────────────────────────────────────

    def _build(self) -> None:
        # ── Shared color scheme selector ──────────────────────────────────────
        scheme_bar = ttk.Frame(self, padding=(6, 4, 6, 0))
        scheme_bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(scheme_bar, text='Color scheme:').pack(side=tk.LEFT, padx=(0, 4))
        self._viz_scheme_var = tk.StringVar(value=list(_COLOR_SCHEMES.keys())[0])
        _scheme_cb = ttk.Combobox(scheme_bar, textvariable=self._viz_scheme_var,
                                  values=list(_COLOR_SCHEMES.keys()),
                                  state='readonly', width=12)
        _scheme_cb.pack(side=tk.LEFT)
        _scheme_cb.bind('<<ComboboxSelected>>', lambda _e: self._on_scheme_change())

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))
        self._nb.bind('<<NotebookTabChanged>>', self._on_tab_change)

        scatter_frame = ttk.Frame(self._nb)
        ridge_frame   = ttk.Frame(self._nb)
        self._nb.add(scatter_frame, text='Scatter')
        self._nb.add(ridge_frame,   text='Ridge')

        self._build_scatter_tab(scatter_frame)
        self._build_ridge_tab(ridge_frame)

    def _build_scatter_tab(self, parent: ttk.Frame) -> None:
        metric_labels = list(self._metric_labels.values())
        feat_names    = [f['name'] for f in self._features]

        # ── Row 1: mode radio buttons ──────────────────────────────────────────
        mode_row = ttk.Frame(parent, padding=(6, 6, 6, 2))
        mode_row.pack(side=tk.TOP, fill=tk.X)

        self._scatter_mode_var = tk.StringVar(value='metric')
        ttk.Radiobutton(
            mode_row, text='Metric vs Metric',
            variable=self._scatter_mode_var, value='metric',
            command=self._on_scatter_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 16))
        ttk.Radiobutton(
            mode_row, text='Band vs Band',
            variable=self._scatter_mode_var, value='band',
            command=self._on_scatter_mode_change,
        ).pack(side=tk.LEFT)

        # ── Row 2: swappable controls ──────────────────────────────────────────
        self._scatter_ctrl_container = ttk.Frame(parent)
        self._scatter_ctrl_container.pack(side=tk.TOP, fill=tk.X)

        # Mode 1: two metric dropdowns + Color by
        self._ctrl_m1 = ttk.Frame(self._scatter_ctrl_container, padding=(6, 0, 6, 2))
        ttk.Label(self._ctrl_m1, text='X:').pack(side=tk.LEFT, padx=(0, 2))
        self._sx_var = tk.StringVar(value=self._metric_labels['band_depth'])
        _cb = ttk.Combobox(self._ctrl_m1, textvariable=self._sx_var, values=metric_labels,
                            state='readonly', width=22)
        _cb.pack(side=tk.LEFT, padx=(0, 10))
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        ttk.Label(self._ctrl_m1, text='Y:').pack(side=tk.LEFT, padx=(0, 2))
        self._sy_var = tk.StringVar(value=self._metric_labels['wl_min'])
        _cb = ttk.Combobox(self._ctrl_m1, textvariable=self._sy_var, values=metric_labels,
                            state='readonly', width=22)
        _cb.pack(side=tk.LEFT, padx=(0, 10))
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        ttk.Label(self._ctrl_m1, text='Color by:').pack(side=tk.LEFT, padx=(0, 2))
        self._sc_var = tk.StringVar(value='Band')
        _cb = ttk.Combobox(self._ctrl_m1, textvariable=self._sc_var,
                            values=['Band', 'Feature group', 'None'],
                            state='readonly', width=13)
        _cb.pack(side=tk.LEFT)
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        self._ctrl_m1.pack(fill=tk.X)   # visible by default

        # Mode 2: one metric + Band A / Band B dropdowns
        self._ctrl_m2 = ttk.Frame(self._scatter_ctrl_container, padding=(6, 0, 6, 2))
        ttk.Label(self._ctrl_m2, text='Metric:').pack(side=tk.LEFT, padx=(0, 2))
        self._sm_var = tk.StringVar(value=self._metric_labels['band_depth'])
        _cb = ttk.Combobox(self._ctrl_m2, textvariable=self._sm_var, values=metric_labels,
                            state='readonly', width=22)
        _cb.pack(side=tk.LEFT, padx=(0, 10))
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        ttk.Label(self._ctrl_m2, text='Band A:').pack(side=tk.LEFT, padx=(0, 2))
        self._sa_var = tk.StringVar(value=feat_names[0] if feat_names else '')
        _cb = ttk.Combobox(self._ctrl_m2, textvariable=self._sa_var, values=feat_names,
                            state='readonly', width=18)
        _cb.pack(side=tk.LEFT, padx=(0, 10))
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        ttk.Label(self._ctrl_m2, text='Band B:').pack(side=tk.LEFT, padx=(0, 2))
        b_default = feat_names[1] if len(feat_names) > 1 else (feat_names[0] if feat_names else '')
        self._sb_var = tk.StringVar(value=b_default)
        _cb = ttk.Combobox(self._ctrl_m2, textvariable=self._sb_var, values=feat_names,
                            state='readonly', width=18)
        _cb.pack(side=tk.LEFT)
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_scatter())
        # _ctrl_m2 stays hidden until user switches mode

        # ── Row 3: clear labels + legend mode ──────────────────────────────────
        btn_row = ttk.Frame(parent, padding=(6, 0, 6, 4))
        btn_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btn_row, text='Clear labels',
                   command=self._clear_scatter_labels).pack(side=tk.LEFT)
        ttk.Separator(btn_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=(12, 8), pady=2)
        ttk.Label(btn_row, text='Legend:').pack(side=tk.LEFT, padx=(0, 4))
        self._scatter_legend_var = tk.StringVar(value='inline')
        for _val, _lbl in [('inline', 'Inline'), ('none', 'None'), ('separate', 'Separate')]:
            ttk.Radiobutton(btn_row, text=_lbl, variable=self._scatter_legend_var,
                            value=_val, command=self._plot_scatter).pack(
                                side=tk.LEFT, padx=(0, 8))

        # ── Band selection (Mode 1 only) ───────────────────────────────────────
        # Container always holds its pack position; inner LF is shown/hidden on mode switch.
        self._scatter_bands_container = ttk.Frame(parent)
        self._scatter_bands_container.pack(side=tk.TOP, fill=tk.X)
        self._scatter_bands_lf = ttk.LabelFrame(self._scatter_bands_container, text='Bands', padding=(4, 2))
        self._scatter_bands_lf.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 0))
        lb_inner = ttk.Frame(self._scatter_bands_lf)
        lb_inner.pack(fill=tk.BOTH)
        lb_vs = ttk.Scrollbar(lb_inner, orient=tk.VERTICAL)
        self._scatter_bands_lb = tk.Listbox(
            lb_inner, selectmode=tk.EXTENDED, height=7,
            exportselection=False, font=('TkFixedFont', 10),
            yscrollcommand=lb_vs.set,
        )
        lb_vs.config(command=self._scatter_bands_lb.yview)
        self._scatter_bands_lb.grid(row=0, column=0, sticky=tk.NSEW)
        lb_vs.grid(row=0, column=1, sticky=tk.NS)
        lb_inner.columnconfigure(0, weight=1)
        for feat in self._features:
            self._scatter_bands_lb.insert(tk.END, feat['name'])
        self._scatter_bands_lb.select_set(0, tk.END)
        self._scatter_bands_lb.bind('<<ListboxSelect>>', lambda _e: self._plot_scatter())
        sel_row = ttk.Frame(self._scatter_bands_lf)
        sel_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(sel_row, text='Select All',
                   command=lambda: (self._scatter_bands_lb.select_set(0, tk.END),
                                    self._plot_scatter())).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_row, text='Clear',
                   command=lambda: (self._scatter_bands_lb.selection_clear(0, tk.END),
                                    self._plot_scatter())).pack(side=tk.LEFT)

        # ── Figure ─────────────────────────────────────────────────────────────
        self._scatter_fig = Figure(figsize=(7, 6), dpi=100)
        self._scatter_ax  = self._scatter_fig.add_subplot(111)
        self._scatter_fig.subplots_adjust(top=0.93, bottom=0.10, left=0.13, right=0.97)

        scatter_canvas = FigureCanvasTkAgg(self._scatter_fig, master=parent)
        NavigationToolbar2Tk(scatter_canvas, parent)
        scatter_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        scatter_canvas.draw()
        self._scatter_canvas = scatter_canvas
        self._scatter_canvas.mpl_connect('button_press_event', self._on_scatter_click)

    def _on_scatter_mode_change(self) -> None:
        if self._scatter_mode_var.get() == 'metric':
            self._ctrl_m2.pack_forget()
            self._ctrl_m1.pack(fill=tk.X)
            # Re-pack inside the container (container itself keeps its position)
            self._scatter_bands_lf.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(2, 0))
        else:
            self._ctrl_m1.pack_forget()
            self._scatter_bands_lf.pack_forget()
            self._ctrl_m2.pack(fill=tk.X)
        self._plot_scatter()

    def _on_tab_change(self, _event=None) -> None:
        if self._nb.index(self._nb.select()) == 1:
            self._plot_ridge()

    def _on_scheme_change(self) -> None:
        tab = self._nb.index(self._nb.select())
        if tab == 0:
            self._plot_scatter()
        else:
            self._plot_ridge()

    def _open_legend_window(self, handles: list, n_cols: int = 1) -> None:
        """Open (or refresh) a detached Toplevel showing only the legend."""
        if self._legend_win is not None:
            try:
                self._legend_win.destroy()
            except tk.TclError:
                pass
            self._legend_win = None

        n_items = len(handles)
        n_rows  = max(1, -(-n_items // n_cols))   # ceiling division
        fig_w   = max(2.5, n_cols * 2.8)
        fig_h   = max(1.2, n_rows * 0.38 + 0.5)

        fig = Figure(figsize=(fig_w, fig_h), dpi=100)
        ax  = fig.add_subplot(111)
        ax.axis('off')
        ax.legend(handles=handles, loc='center', fontsize=9,
                  ncols=n_cols, framealpha=0.9, markerscale=1.3)
        fig.tight_layout(pad=0.3)

        win = tk.Toplevel(self)
        win.title('Legend')
        win.resizable(True, True)
        canvas = FigureCanvasTkAgg(fig, master=win)
        NavigationToolbar2Tk(canvas, win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        win.geometry(f'{int(fig_w * 100 + 20)}x{int(fig_h * 100 + 56)}')
        self._legend_win = win

    def _build_ridge_tab(self, parent: ttk.Frame) -> None:
        metric_labels = list(self._metric_labels.values())

        ctrl = ttk.Frame(parent, padding=(6, 6, 6, 0))
        ctrl.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(ctrl, text='Metric:').pack(side=tk.LEFT, padx=(0, 2))
        self._vm_var = tk.StringVar(value=self._metric_labels['band_depth'])
        _cb = ttk.Combobox(ctrl, textvariable=self._vm_var, values=metric_labels,
                            state='readonly', width=24)
        _cb.pack(side=tk.LEFT, padx=(0, 10))
        _cb.bind('<<ComboboxSelected>>', lambda _e: self._plot_ridge())
        self._vp_var = tk.BooleanVar(value=True)
        self._vp_var.trace_add('write', lambda *_: self._plot_ridge())
        ttk.Checkbutton(ctrl, text='Show points',
                        variable=self._vp_var).pack(side=tk.LEFT, padx=(0, 10))

        lb_outer = ttk.LabelFrame(parent, text='Bands', padding=(4, 2))
        lb_outer.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 0))
        lb_inner = ttk.Frame(lb_outer)
        lb_inner.pack(fill=tk.BOTH)
        lb_vs = ttk.Scrollbar(lb_inner, orient=tk.VERTICAL)
        self._ridge_lb = tk.Listbox(
            lb_inner, selectmode=tk.EXTENDED, height=7,
            exportselection=False, font=('TkFixedFont', 10),
            yscrollcommand=lb_vs.set,
        )
        lb_vs.config(command=self._ridge_lb.yview)
        self._ridge_lb.grid(row=0, column=0, sticky=tk.NSEW)
        lb_vs.grid(row=0, column=1, sticky=tk.NS)
        lb_inner.columnconfigure(0, weight=1)
        for feat in self._features:
            self._ridge_lb.insert(tk.END, feat['name'])
        self._ridge_lb.select_set(0, tk.END)
        self._ridge_lb.bind('<<ListboxSelect>>', lambda _e: self._plot_ridge())

        sel_row = ttk.Frame(lb_outer)
        sel_row.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(sel_row, text='Select All',
                   command=lambda: (self._ridge_lb.select_set(0, tk.END),
                                    self._plot_ridge())).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(sel_row, text='Clear',
                   command=lambda: (self._ridge_lb.selection_clear(0, tk.END),
                                    self._plot_ridge())).pack(side=tk.LEFT)

        self._ridge_fig = Figure(figsize=(7, 5), dpi=100)
        self._ridge_ax  = self._ridge_fig.add_subplot(111)
        self._ridge_fig.subplots_adjust(top=0.93, bottom=0.08, left=0.28, right=0.97)

        violin_canvas = FigureCanvasTkAgg(self._ridge_fig, master=parent)
        NavigationToolbar2Tk(violin_canvas, parent)
        violin_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        violin_canvas.draw()
        self._ridge_canvas = violin_canvas

    # ── Scatter plotting ──────────────────────────────────────────────────────

    def _plot_scatter(self) -> None:
        ax = self._scatter_ax
        ax.cla()
        self._clear_scatter_labels(redraw=False)
        if self._scatter_mode_var.get() == 'metric':
            self._plot_scatter_metric(ax)
        else:
            self._plot_scatter_band(ax)
        self._scatter_canvas.draw_idle()

    def _plot_scatter_metric(self, ax) -> None:
        """Mode 1: (spectrum × band) pairs; X and Y are independent metrics."""
        x_key = self._label_to_key.get(self._sx_var.get())
        y_key = self._label_to_key.get(self._sy_var.get())
        if x_key is None or y_key is None:
            return

        sel_idx = list(self._scatter_bands_lb.curselection())
        selected_feats = {self._features[i]['name'] for i in sel_idx}
        valid = [r for r in self._records
                 if r['feature'] in selected_feats
                 and not (np.isnan(r.get(x_key, np.nan)) or
                          np.isnan(r.get(y_key, np.nan)))]
        if not valid:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            return

        color_by = self._sc_var.get()
        _palette = _COLOR_SCHEMES[self._viz_scheme_var.get()]

        _src_markers: dict[str, str] = {'Data': 'o', 'Library': '^', '': 'o'}
        mixed_sources = len({r.get('source', '') for r in valid} - {''}) > 1

        def _scatter_pts(pts: list[dict], color) -> None:
            """Draw pts split by source so marker shapes are correct; no labels."""
            for src, mkr in _src_markers.items():
                sub = [r for r in pts if r.get('source', '') == src]
                if sub:
                    ax.scatter([r[x_key] for r in sub], [r[y_key] for r in sub],
                               marker=mkr, s=30, alpha=0.75, color=color)

        color_handles: list = []
        n_cols = 1

        if color_by == 'Band':
            feats   = [f['name'] for f in self._features]
            f_color = {f: _palette[i % len(_palette)] for i, f in enumerate(feats)}
            for f in feats:
                pts = [r for r in valid if r['feature'] == f]
                if pts:
                    _scatter_pts(pts, f_color[f])
                    color_handles.append(
                        Line2D([0], [0], marker='s', linestyle='none',
                               color=f_color[f], markersize=7, label=f)
                    )
            n_cols = max(1, len(color_handles) // 12)
        elif color_by == 'Feature group':
            groups  = sorted(set(r['group'] for r in valid))
            g_color = {g: _dark2[i % len(_dark2)] for i, g in enumerate(groups)}
            for g in groups:
                pts = [r for r in valid if r['group'] == g]
                _scatter_pts(pts, g_color[g])
                color_handles.append(
                    Line2D([0], [0], marker='s', linestyle='none',
                           color=g_color[g], markersize=7,
                           label=g if g else '(none)')
                )
        else:
            _scatter_pts(valid, 'steelblue')

        # Prepend source-shape handles when both Data and Library are present
        if mixed_sources:
            src_handles = [
                Line2D([0], [0], marker='o', linestyle='none', color='gray',
                       markersize=6, label='Data'),
                Line2D([0], [0], marker='^', linestyle='none', color='gray',
                       markersize=6, label='Library'),
            ]
            all_handles = src_handles + color_handles
        else:
            all_handles = color_handles

        self._scatter_legend_handles = all_handles

        if all_handles:
            legend_mode = self._scatter_legend_var.get()
            if legend_mode == 'inline':
                ax.legend(handles=all_handles, fontsize=8, markerscale=1.1,
                          loc='best', framealpha=0.7, ncols=n_cols)
            elif legend_mode == 'separate':
                self._open_legend_window(all_handles, n_cols)

        self._scatter_xdata = np.array([r[x_key] for r in valid])
        self._scatter_ydata = np.array([r[y_key] for r in valid])
        self._scatter_names = [f"{r['spectrum']}  [{r['feature']}]" for r in valid]

        ax.set_xlabel(self._sx_var.get(), fontsize=11)
        ax.set_ylabel(self._sy_var.get(), fontsize=11)
        ax.set_title(f"{self._sx_var.get()} vs {self._sy_var.get()}", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _plot_scatter_band(self, ax) -> None:
        """Mode 2: one point per spectrum; X = Band A metric, Y = Band B metric."""
        metric_key = self._label_to_key.get(self._sm_var.get())
        band_a     = self._sa_var.get()
        band_b     = self._sb_var.get()
        if metric_key is None or not band_a or not band_b:
            return

        # Aggregate per-spectrum per-band values (include source)
        sp_vals:   dict[str, dict[str, float]] = {}
        sp_source: dict[str, str]              = {}
        for r in self._records:
            sp_vals.setdefault(r['spectrum'], {})[r['feature']] = r.get(metric_key, np.nan)
            sp_source[r['spectrum']] = r.get('source', '')

        xs, ys, names, srcs = [], [], [], []
        for sp, fv in sp_vals.items():
            x = fv.get(band_a, np.nan)
            y = fv.get(band_b, np.nan)
            if not np.isnan(x) and not np.isnan(y):
                xs.append(float(x))
                ys.append(float(y))
                names.append(sp)
                srcs.append(sp_source.get(sp, ''))

        if not xs:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            return

        xs_arr = np.array(xs)
        ys_arr = np.array(ys)
        srcs_arr = np.array(srcs)
        _src_markers = {'Data': 'o', 'Library': '^', '': 'o'}
        mixed_sources = len(set(srcs) - {''}) > 1
        for src, mkr in _src_markers.items():
            idx = np.where(srcs_arr == src)[0]
            if not len(idx):
                continue
            lbl = src if (mixed_sources and src) else None
            ax.scatter(xs_arr[idx], ys_arr[idx],
                       marker=mkr, s=35, alpha=0.75, color='steelblue', label=lbl)
        if mixed_sources:
            ax.legend(fontsize=8, markerscale=1.1, loc='best', framealpha=0.7)

        self._scatter_xdata = xs_arr
        self._scatter_ydata = ys_arr
        self._scatter_names = names

        metric_label = self._sm_var.get()
        ax.set_xlabel(f'{band_a}  —  {metric_label}', fontsize=11)
        ax.set_ylabel(f'{band_b}  —  {metric_label}', fontsize=11)
        ax.set_title(f'{metric_label}: {band_a} vs {band_b}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.3)

    def _on_scatter_click(self, event) -> None:
        if event.inaxes != self._scatter_ax:
            return
        if self._scatter_canvas.toolbar.mode != '':
            return
        if self._scatter_xdata is None or len(self._scatter_xdata) == 0:
            return

        xy_disp    = self._scatter_ax.transData.transform(
            np.column_stack([self._scatter_xdata, self._scatter_ydata]))
        click_disp = np.array([event.x, event.y])
        dists      = np.linalg.norm(xy_disp - click_disp, axis=1)
        idx        = int(np.argmin(dists))
        if dists[idx] > 15:
            return

        name = self._scatter_names[idx]
        if name in self._scatter_annotations:
            self._scatter_annotations.pop(name).remove()
        else:
            ann = self._scatter_ax.annotate(
                name,
                xy=(self._scatter_xdata[idx], self._scatter_ydata[idx]),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8, clip_on=True,
                bbox=dict(boxstyle='round,pad=0.25',
                          fc='lightyellow', alpha=0.85, lw=0.5),
            )
            self._scatter_annotations[name] = ann
        self._scatter_canvas.draw_idle()

    def _clear_scatter_labels(self, redraw: bool = True) -> None:
        for ann in list(self._scatter_annotations.values()):
            ann.remove()
        self._scatter_annotations.clear()
        if redraw:
            self._scatter_canvas.draw_idle()

    # ── Ridge plotting ────────────────────────────────────────────────────────

    def _plot_ridge(self) -> None:
        from matplotlib.patches import Patch
        ax = self._ridge_ax
        ax.cla()

        metric_key = self._label_to_key.get(self._vm_var.get())
        if metric_key is None:
            self._ridge_canvas.draw_idle()
            return

        sel_idx = list(self._ridge_lb.curselection())
        if not sel_idx:
            messagebox.showinfo('Ridge Plot', 'Select at least one band.', parent=self)
            return

        feat_names = [self._features[i]['name'] for i in sel_idx]

        # Detect whether both Data and Library are present in records
        all_rec_sources = {r.get('source', '') for r in self._records} - {''}
        mixed_sources   = len(all_rec_sources) > 1

        # Build per-band value arrays, split by source when mixed
        plot_names: list[str]        = []
        plot_d:     list[np.ndarray] = []   # Data vals (or all vals when not mixed)
        plot_l:     list[np.ndarray] = []   # Library vals (empty when not mixed)

        for name in feat_names:
            recs = [r for r in self._records
                    if r['feature'] == name
                    and not np.isnan(r.get(metric_key, np.nan))]
            all_arr = np.array([r[metric_key] for r in recs])
            if len(all_arr) < 2:
                continue
            if mixed_sources:
                d_arr = np.array([r[metric_key] for r in recs if r.get('source') == 'Data'])
                l_arr = np.array([r[metric_key] for r in recs if r.get('source') == 'Library'])
            else:
                d_arr = all_arr
                l_arr = np.array([])
            plot_names.append(name)
            plot_d.append(d_arr)
            plot_l.append(l_arr)

        if not plot_names:
            ax.text(0.5, 0.5,
                    'Insufficient data\n(need ≥ 2 spectra per band)',
                    transform=ax.transAxes, ha='center', va='center', color='gray')
            self._ridge_canvas.draw_idle()
            return

        n            = len(plot_names)
        scheme       = _COLOR_SCHEMES[self._viz_scheme_var.get()]
        colors       = [scheme[i % len(scheme)] for i in range(n)]
        band_spacing = 1.0
        kde_height   = 0.70
        cloud_height = 0.28

        all_chunks = [v for v in plot_d + plot_l if len(v)]
        all_vals   = np.concatenate(all_chunks)
        x_margin   = max((all_vals.max() - all_vals.min()) * 0.08, 1e-6)
        x_range    = np.linspace(all_vals.min() - x_margin,
                                 all_vals.max() + x_margin, 400)

        rng      = np.random.default_rng(seed=0)
        show_pts = self._vp_var.get()

        def _draw_kde(vals: np.ndarray, baseline: float, color,
                      hatch: str | None = None) -> None:
            if len(vals) < 2:
                return
            kde = gaussian_kde(vals, bw_method='scott')
            ky  = kde(x_range)
            ky  = ky / ky.max() * kde_height
            if hatch:
                ax.fill_between(x_range, baseline, baseline + ky,
                                color=color, alpha=0.18, hatch=hatch, zorder=2)
                ax.plot(x_range, baseline + ky,
                        color=color, linewidth=1.4, linestyle='--', alpha=0.85, zorder=3)
            else:
                ax.fill_between(x_range, baseline, baseline + ky,
                                color=color, alpha=0.45, zorder=2)
                ax.plot(x_range, baseline + ky,
                        color=color, linewidth=1.4, alpha=0.9, zorder=3)

        baselines: list[float] = []
        for i, (name, d_vals, l_vals, color) in enumerate(
                zip(plot_names, plot_d, plot_l, colors)):
            baseline = float((n - 1 - i) * band_spacing)
            baselines.append(baseline)

            ax.axhline(baseline, color=color, linewidth=0.6, alpha=0.35, zorder=1)

            if mixed_sources:
                _draw_kde(d_vals, baseline, color, hatch=None)      # Data: solid
                _draw_kde(l_vals, baseline, color, hatch='///')      # Library: hatched
                if show_pts:
                    if len(d_vals):
                        jitter = rng.uniform(-cloud_height, 0.0, len(d_vals))
                        ax.scatter(d_vals, baseline + jitter,
                                   s=4, marker='o', alpha=0.5, color=color,
                                   linewidths=0, zorder=4)
                    if len(l_vals):
                        jitter = rng.uniform(-cloud_height, 0.0, len(l_vals))
                        ax.scatter(l_vals, baseline + jitter,
                                   s=9, marker='^', alpha=0.5, color=color,
                                   linewidths=0, zorder=4)
            else:
                _draw_kde(d_vals, baseline, color, hatch=None)
                if show_pts:
                    jitter = rng.uniform(-cloud_height, 0.0, len(d_vals))
                    ax.scatter(d_vals, baseline + jitter,
                               s=4, alpha=0.45, color=color,
                               linewidths=0, zorder=4)

        ax.set_yticks(baselines)
        ax.set_yticklabels(plot_names, fontsize=9)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-cloud_height - 0.15,
                    (n - 1) * band_spacing + kde_height + 0.15)

        ax.set_xlabel(self._vm_var.get(), fontsize=11)
        ax.set_title(f'Distribution of {self._vm_var.get()} by band', fontsize=11)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ('left', 'right', 'top'):
            ax.spines[spine].set_visible(False)

        if mixed_sources:
            legend_handles = [
                Patch(facecolor='gray', alpha=0.55, label='Data'),
                Patch(facecolor='gray', alpha=0.25, hatch='///', label='Library'),
            ]
            ax.legend(handles=legend_handles, fontsize=8,
                      loc='upper right', framealpha=0.7)

        n_skipped = len(feat_names) - len(plot_names)
        if n_skipped:
            ax.text(0.99, 0.99, f'{n_skipped} band(s) skipped (< 2 spectra)',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='gray')

        self._ridge_canvas.draw_idle()


class BandResultsWindow(tk.Toplevel):
    """
    Non-modal window displaying a band-parameter table for all plotted spectra.

    Rows are spectra (data then library); columns are all nine band parameters
    for each selected feature.  An "Export CSV" button saves the table.
    """

    def __init__(
        self,
        master:   tk.Misc,
        features: list[dict],
        results:  dict[str, dict],
        unit:     str = 'nm',
    ) -> None:
        super().__init__(master)
        self.title('Band Parameters — Results')
        self.resizable(True, True)

        self._features = features
        self._results  = results   # {spectrum_name: {feat_name: dict | None}}
        self._unit     = unit
        self._build()

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=(8, 6))
        frm.pack(fill=tk.BOTH, expand=True)

        scale = 1e-3 if self._unit == 'µm' else 1.0

        # ── Build column spec ─────────────────────────────────────────────
        col_ids: list[str]           = ['spectrum']
        col_hdg: dict[str, str]      = {'spectrum': 'Spectrum'}
        for i, feat in enumerate(self._features):
            abbr = feat['name'][:16]
            for key, tmpl, scaled, _ in _BAND_METRIC_SPEC:
                cid   = f'f{i}_{key}'
                lbl   = tmpl.format(unit=self._unit) if scaled else tmpl
                col_ids.append(cid)
                col_hdg[cid] = f'{abbr}\n{lbl}'

        # ── Treeview ──────────────────────────────────────────────────────
        tv_frame = ttk.Frame(frm)
        tv_frame.pack(fill=tk.BOTH, expand=True)

        tv = ttk.Treeview(tv_frame, columns=col_ids, show='headings',
                          height=min(20, len(self._results) + 1))
        xsb = ttk.Scrollbar(tv_frame, orient=tk.HORIZONTAL, command=tv.xview)
        ysb = ttk.Scrollbar(tv_frame, orient=tk.VERTICAL,   command=tv.yview)
        tv.configure(xscrollcommand=xsb.set, yscrollcommand=ysb.set)

        tv.column('spectrum', width=160, anchor='w', stretch=False)
        tv.heading('spectrum', text='Spectrum')
        for cid in col_ids[1:]:
            tv.column(cid,  width=100, anchor='center', stretch=False)
            tv.heading(cid, text=col_hdg[cid])

        xsb.pack(side=tk.BOTTOM, fill=tk.X)
        ysb.pack(side=tk.RIGHT,  fill=tk.Y)
        tv.pack(side=tk.LEFT,    fill=tk.BOTH, expand=True)
        self._tv = tv

        # ── Populate rows ─────────────────────────────────────────────────
        for sp_name, feat_results in self._results.items():
            row_vals: list[str] = [sp_name]
            for feat in self._features:
                bp = feat_results.get(feat['name'])
                if bp is None:
                    row_vals += ['—'] * len(_BAND_METRIC_SPEC)
                else:
                    for key, _, scaled, fmt in _BAND_METRIC_SPEC:
                        val = bp.get(key)
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            row_vals.append('—')
                        else:
                            v = val * scale if scaled else val
                            row_vals.append(format(v, fmt))
            tv.insert('', tk.END, values=row_vals)

        # ── Action buttons ────────────────────────────────────────────────
        ttk.Separator(frm, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(6, 4))
        btn_row = ttk.Frame(frm)
        btn_row.pack(anchor='e')
        ttk.Button(
            btn_row, text='Visualize…', command=self._on_visualize,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_row, text='Export', command=self._export,
        ).pack(side=tk.LEFT)

    def _on_visualize(self) -> None:
        BandVizWindow(self, self._features, self._results, self._unit)

    def _export(self) -> None:
        from tkinter import filedialog
        import csv
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export band parameters',
        )
        if not path:
            return
        scale = 1e-3 if self._unit == 'µm' else 1.0
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            # Header row
            header = ['Spectrum']
            for feat in self._features:
                n = feat['name']
                for key, tmpl, scaled, _ in _BAND_METRIC_SPEC:
                    lbl = tmpl.format(unit=self._unit) if scaled else tmpl
                    header.append(f'{n} {lbl.replace(chr(10), " ")}')
            writer.writerow(header)
            # Data rows
            for sp_name, feat_results in self._results.items():
                row: list = [sp_name]
                for feat in self._features:
                    bp = feat_results.get(feat['name'])
                    if bp is None:
                        row += [''] * len(_BAND_METRIC_SPEC)
                    else:
                        for key, _, scaled, _ in _BAND_METRIC_SPEC:
                            val = bp.get(key)
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                row.append('')
                            else:
                                row.append(val * scale if scaled else val)
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Load format dialog
# ---------------------------------------------------------------------------

class LoadFormatDialog(tk.Toplevel):
    """
    Modal dialog for choosing the reflectance data file format before loading.

    After ``wait_window`` returns, inspect ``self.result``:

    * ``'csv'``  — generic comma-separated reflectance file
    * ``'asd'``  — ASD spectrometer tab-separated text export
    * ``None``   — user cancelled

    Attributes
    ----------
    result : str or None
        Selected format tag, or None if the dialog was dismissed.
    """

    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title('Load Data')
        self.resizable(False, False)
        self.result: str | None = None

        self._fmt_var = tk.StringVar(value='csv')

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill='both', expand=True)

        ttk.Label(frame, text='Select file format:').pack(anchor='w', pady=(0, 8))
        ttk.Radiobutton(frame, text='Generic CSV (.csv)',
                        variable=self._fmt_var, value='csv').pack(anchor='w')
        ttk.Radiobutton(frame, text='ASD Text (.txt)',
                        variable=self._fmt_var, value='asd').pack(anchor='w', pady=(4, 0))

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill='x', pady=(14, 0))
        ttk.Button(btn_frame, text='Load',
                   command=self._on_load).pack(side='right', padx=(6, 0))
        ttk.Button(btn_frame, text='Cancel',
                   command=self.destroy).pack(side='right')

        self.transient(parent)
        self.grab_set()
        self.wait_window()

    def _on_load(self) -> None:
        self.result = self._fmt_var.get()
        self.destroy()


# ---------------------------------------------------------------------------
# ASD white reference correction dialog
# ---------------------------------------------------------------------------

class WhiteReferenceCorrectionDialog(tk.Toplevel):
    """
    Modal dialog for assigning white references and applying reflectance
    correction to ASD spectra before they enter the data pool.

    Each spectrum is listed in file order.  Those whose mean reflectance
    exceeds ``_WR_THRESHOLD`` are auto-detected as white references and
    shown greyed out with their "White ref" checkbox ticked.  The user may
    override any assignment, then for each sample spectrum supply a name
    and choose which white reference to use.

    Correction applied: ``corrected = sample × white_reference``

    After ``wait_window`` returns, check ``self.cancelled``.  If ``False``,
    ``self.result`` holds one entry per corrected sample.

    Attributes
    ----------
    cancelled : bool
    result : list[dict]
        Each entry: ``{'name': str, 'xaxis': np.ndarray, 'data': np.ndarray}``.
    """

    _WR_THRESHOLD: float = 0.75

    # Matches a 5-digit sequence + optional file-extension suffix + any
    # of the supported separators (=  ,  :  tab) + the sample name.
    _NAME_FILE_RE = re.compile(
        r'.*?(\d{5})(?:\.\w+)*\s*[=,:\t]\s*(.+)',
        re.IGNORECASE,
    )

    def __init__(
        self,
        master:      tk.Misc,
        xaxis:       np.ndarray,
        spectra:     dict[str, np.ndarray],
        source_path: Path | None = None,
    ) -> None:
        super().__init__(master)
        self.title('ASD White Reference Correction')
        self.resizable(False, True)
        self.grab_set()
        self.transient(master)
        self.minsize(780, 420)

        self.cancelled: bool       = True
        self.result:    list[dict] = []

        self._xaxis       = xaxis
        self._spectra     = list(spectra.items())   # [(orig_name, data), …] in order
        self._source_path = source_path

        self._initial_is_wr: list[bool] = [
            bool(np.mean(data) > self._WR_THRESHOLD)
            for _, data in self._spectra
        ]

        self._rows:          list[dict]    = []
        self._save_csv_var:  tk.BooleanVar = tk.BooleanVar(value=True)

        self._build()
        self.wait_window(self)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_wr_names(self) -> list[str]:
        return [r['orig_name'] for r in self._rows if r['is_wr_var'].get()]

    def _preceding_wr_name(self, idx: int) -> str:
        """Return the most recent WR name before *idx* (current state), or first WR."""
        for i in range(idx - 1, -1, -1):
            if self._rows[i]['is_wr_var'].get():
                return self._rows[i]['orig_name']
        for r in self._rows:
            if r['is_wr_var'].get():
                return r['orig_name']
        return ''

    def _refresh_wr_choices(self) -> None:
        wr_names = self._get_wr_names()
        for row in self._rows:
            if row['is_wr_var'].get():
                continue
            cb = row['wr_cb']
            current = row['wr_sel_var'].get()
            cb.config(values=wr_names)
            if wr_names and current not in wr_names:
                row['wr_sel_var'].set(wr_names[0])
            elif not wr_names:
                row['wr_sel_var'].set('')

    # ── Name-file loading ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_name_file(path: Path) -> dict[str, str]:
        """
        Parse a free-form notes file that maps sequence numbers to sample names.

        Recognised formats (separator may be ``=``, ``,``, ``:``, or tab;
        spaces around the separator are ignored):

        * ``00000 = WR1``
        * ``filename_00000.asd.sco, Sample A``
        * ``00000: S10 M1``

        Lines that do not contain a 5-digit key followed by a separator are
        silently treated as comments.
        """
        mapping: dict[str, str] = {}
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            m = WhiteReferenceCorrectionDialog._NAME_FILE_RE.match(line.rstrip())
            if m:
                seq, name = m.group(1), m.group(2).strip()
                if name:
                    mapping[seq] = name
        return mapping

    @staticmethod
    def _extract_seq(orig_name: str) -> str | None:
        """Extract the 5-digit sequence number from an ASD column name."""
        m = re.search(r'(\d{5})\.asd', orig_name, re.IGNORECASE)
        if m:
            return m.group(1)
        # Fallback: last standalone 5-digit group in the string
        matches = re.findall(r'(?<!\d)(\d{5})(?!\d)', orig_name)
        return matches[-1] if matches else None

    def _apply_name_file(self, mapping: dict[str, str]) -> int:
        """
        Populate ``name_var`` for every row whose sequence number appears in
        *mapping*.  WR status is not changed — names only.

        Returns the number of rows that were updated.
        """
        n = 0
        for row in self._rows:
            seq = self._extract_seq(row['orig_name'])
            if seq is None or seq not in mapping:
                continue
            row['name_var'].set(mapping[seq])
            n += 1
        return n

    def _on_load_names(self) -> None:
        path = filedialog.askopenfilename(
            title='Load name mapping',
            filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
            parent=self,
        )
        if not path:
            return
        mapping = self._parse_name_file(Path(path))
        n = self._apply_name_file(mapping)
        self._names_label.config(
            text=f'{n} / {len(self._rows)} spectra matched',
            foreground='' if n else 'red',
        )

    # ── Toggle callback ───────────────────────────────────────────────────────

    def _on_wr_toggle(self, idx: int) -> None:
        row   = self._rows[idx]
        is_wr = row['is_wr_var'].get()

        row['orig_label'].config(foreground='gray' if is_wr else '')
        row['name_entry'].config(state='disabled' if is_wr else 'normal')
        row['wr_cb'].config(state='disabled' if is_wr else 'readonly')

        if not is_wr:
            default = self._preceding_wr_name(idx)
            if default and row['wr_sel_var'].get() not in self._get_wr_names():
                row['wr_sel_var'].set(default)

        self._refresh_wr_choices()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=(12, 10))
        outer.pack(fill=tk.BOTH, expand=True)

        n_wr = sum(self._initial_is_wr)
        wr_note = (f'{n_wr} spectrum{"a" if n_wr != 1 else ""} auto-detected as '
                   f'white reference{"s" if n_wr != 1 else ""} (mean reflectance > '
                   f'{self._WR_THRESHOLD}).  Uncheck "White ref" to treat as a sample '
                   f'instead.  For each sample, enter a name and select which white '
                   f'reference to apply.\n'
                   f'Correction: reflectance = sample × white_reference')
        ttk.Label(
            outer, text=wr_note, wraplength=720, justify='left',
        ).pack(anchor='w', pady=(0, 4))

        # ── Load names from file ──────────────────────────────────────────────
        names_row = ttk.Frame(outer)
        names_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(
            names_row, text='Load names from file…', command=self._on_load_names,
        ).pack(side=tk.LEFT)
        self._names_label = ttk.Label(names_row, text='', foreground='gray')
        self._names_label.pack(side=tk.LEFT, padx=(8, 0))

        # ── Column headers ────────────────────────────────────────────────────
        hdr = ttk.Frame(outer)
        hdr.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hdr, text='Original name',  width=32, anchor='w'     ).pack(side=tk.LEFT)
        ttk.Label(hdr, text='White ref',      width=9,  anchor='center').pack(side=tk.LEFT)
        ttk.Label(hdr, text='Sample name',    width=26, anchor='w'     ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(hdr, text='Use white ref',  width=26, anchor='w'     ).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 4))

        # ── Scrollable rows ───────────────────────────────────────────────────
        scroll_outer = ttk.Frame(outer)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        n      = len(self._spectra)
        height = min(420, n * 27 + 20)
        canvas = tk.Canvas(scroll_outer, borderwidth=0, highlightthickness=0,
                           height=height)
        vsb = ttk.Scrollbar(scroll_outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        rows_frame = ttk.Frame(canvas)
        cwin = canvas.create_window((0, 0), window=rows_frame, anchor='nw')
        rows_frame.bind('<Configure>',
                        lambda _e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(cwin, width=e.width))

        wr_names_init = [name for i, (name, _) in enumerate(self._spectra)
                         if self._initial_is_wr[i]]

        for idx, (orig_name, data) in enumerate(self._spectra):
            is_wr_init = self._initial_is_wr[idx]

            is_wr_var  = tk.BooleanVar(value=is_wr_init)
            name_var   = tk.StringVar(value=orig_name)
            wr_sel_var = tk.StringVar(value='')

            row_frm = ttk.Frame(rows_frame)
            row_frm.pack(fill=tk.X, pady=1)

            fg = 'gray' if is_wr_init else ''
            orig_label = ttk.Label(row_frm, text=orig_name, width=32, anchor='w',
                                   foreground=fg)
            orig_label.pack(side=tk.LEFT)

            ttk.Checkbutton(
                row_frm, variable=is_wr_var,
                command=lambda i=idx: self._on_wr_toggle(i),
            ).pack(side=tk.LEFT, padx=(6, 2))

            name_entry = ttk.Entry(
                row_frm, textvariable=name_var, width=26,
                state='disabled' if is_wr_init else 'normal',
            )
            name_entry.pack(side=tk.LEFT, padx=(6, 0))

            wr_cb = ttk.Combobox(
                row_frm, textvariable=wr_sel_var, width=26,
                values=wr_names_init,
                state='disabled' if is_wr_init else 'readonly',
            )
            wr_cb.pack(side=tk.LEFT, padx=(6, 0))

            self._rows.append({
                'orig_name': orig_name,
                'data':      data,
                'is_wr_var': is_wr_var,
                'name_var':  name_var,
                'wr_sel_var': wr_sel_var,
                'orig_label': orig_label,
                'name_entry': name_entry,
                'wr_cb':      wr_cb,
            })

        # Set default WR selections (preceding WR rule) for sample rows.
        for idx, row in enumerate(self._rows):
            if row['is_wr_var'].get() or not wr_names_init:
                continue
            default = ''
            for i in range(idx - 1, -1, -1):
                if self._initial_is_wr[i]:
                    default = self._spectra[i][0]
                    break
            row['wr_sel_var'].set(default if default else wr_names_init[0])

        # ── Buttons ───────────────────────────────────────────────────────────
        ttk.Separator(outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(8, 6))
        btn_row = ttk.Frame(outer)
        btn_row.pack()
        ttk.Checkbutton(
            btn_row, text='Save to CSV', variable=self._save_csv_var,
        ).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Button(btn_row, text='Apply',  command=self._apply).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT)

    # ── Apply ─────────────────────────────────────────────────────────────────

    def _apply(self) -> None:
        wr_lookup = {r['orig_name']: r['data']
                     for r in self._rows if r['is_wr_var'].get()}

        if not wr_lookup:
            messagebox.showwarning(
                'No white references',
                'Mark at least one spectrum as a white reference.',
                parent=self,
            )
            return

        sample_rows = [r for r in self._rows if not r['is_wr_var'].get()]
        if not sample_rows:
            messagebox.showwarning(
                'No samples',
                'All spectra are marked as white references — nothing to correct.',
                parent=self,
            )
            return

        for r in sample_rows:
            name = r['name_var'].get().strip()
            if not name:
                messagebox.showerror(
                    'Missing sample name',
                    f"Enter a name for spectrum '{r['orig_name']}'.",
                    parent=self,
                )
                return
            wr_name = r['wr_sel_var'].get()
            if not wr_name or wr_name not in wr_lookup:
                messagebox.showerror(
                    'No white reference selected',
                    f"Select a white reference for '{r['orig_name']}'.",
                    parent=self,
                )
                return

        self.result = [
            {
                'name':  r['name_var'].get().strip(),
                'xaxis': self._xaxis,
                'data':  r['data'] * wr_lookup[r['wr_sel_var'].get()],
            }
            for r in sample_rows
        ]

        if self._save_csv_var.get() and self._source_path is not None:
            saveReflectanceCSV(
                self._source_path.with_suffix('.csv'),
                self._xaxis,
                {item['name']: item['data'] for item in self.result},
            )

        self.cancelled = False
        self.destroy()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ReflectanceVSWIR(tk.Tk):
    """
    Top-level application window for VSWIR reflectance data processing.

    Attributes
    ----------
    _data_pool : list[dict]
        All spectra loaded from data CSVs; shown in the left listbox.
        Each entry: ``{name, xaxis, data, source}``.
    _data_plot_items : list[dict]
        Plot entries from the data side, built by the Add buttons.
        Each entry adds one slot to the top of the plot stack.
        Entry schema: ``{kind, name, xaxis, data, source}`` where *kind* is
        ``'single'`` (data is 1-D) or ``'group'`` (data is 2-D, n_spec × n_ch).
    _library : dict
        Spectral library keyed by column name; empty until a library is loaded.
        Each entry: ``{name, xaxis, data, source}``.
    _lib_plot_items : list[dict]
        Plot entries from the library side, built by the Add buttons.
        Each entry adds one slot to the bottom of the plot stack (prepended).
        Same schema as _data_plot_items.
    _filtered_ids : list
        Column names currently visible in the library listbox after filtering.
    _filter_key_vars : list[tk.StringVar]
        One per filter row — which metadata field to filter on.
    _filter_val_vars : list[tk.StringVar]
        One per filter row — which value to match within that field.
    _filter_key_combos : list[ttk.Combobox]
        Keyword-selector widgets (left Combobox in each filter row).
    _filter_val_combos : list[ttk.Combobox]
        Value-selector widgets (right Combobox in each filter row).
    _data_offset_var : tk.DoubleVar
        Vertical increment between consecutive data plot entries.
    _lib_offset_var : tk.DoubleVar
        Vertical increment between consecutive library plot entries.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title(f'ReflectanceVSWIR  v{__version__}')
        self.geometry('1400x900')
        self.minsize(1000, 660)

        # ── Data state ────────────────────────────────────────────────────────
        self._data_pool:       list[dict] = []   # loaded, shown in left listbox
        self._data_plot_items: list[dict] = []   # actively plotted (data top)
        self._library:         dict       = {}   # loaded, shown in right listbox
        self._lib_plot_items:  list[dict] = []   # actively plotted (library bottom)
        self._filtered_ids:    list       = []

        # ── Offset controls ───────────────────────────────────────────────────
        self._data_offset_var    = tk.DoubleVar(value=0.1)
        self._lib_offset_var     = tk.DoubleVar(value=0.1)
        self._section_gap_var    = tk.DoubleVar(value=0.5)

        # ── Plot display options ──────────────────────────────────────────────
        self._show_labels_var    = tk.BooleanVar(value=False)
        self._xaxis_unit_var     = tk.StringVar(value='nm')
        self._xaxis_min_var      = tk.DoubleVar(value=350.0)
        self._xaxis_max_var      = tk.DoubleVar(value=2500.0)
        self._show_cr_var        = tk.BooleanVar(value=False)
        self._cr_available:  bool                      = False
        self._cr_wl_range:   tuple[float, float] | None = None
        # Widget refs for enabling/disabling after CR is computed
        self._btn_remove_cr:      ttk.Button     | None = None
        self._cb_show_cr:         ttk.Checkbutton| None = None
        self._xaxis_min_sb:       ttk.Spinbox    | None = None
        self._xaxis_max_sb:       ttk.Spinbox    | None = None
        self._xaxis_unit_label:   ttk.Label      | None = None

        # ── Smoothing state ───────────────────────────────────────────────────
        self._show_smoothed_var  = tk.BooleanVar(value=False)
        self._smooth_available:  bool        = False
        self._smooth_params:     dict | None = None
        self._btn_smooth:        ttk.Button      | None = None
        self._cb_show_smoothed:  ttk.Checkbutton | None = None

        # ── Feature markers ───────────────────────────────────────────────────
        self._preset_data:    list[dict]      = self._load_preset_features()
        # Per-preset state: {name: {enabled: BooleanVar, wl_var: DoubleVar, fwhm_var: DoubleVar}}
        self._preset_vars:    dict[str, dict] = {
            p['name']: {
                'enabled':  tk.BooleanVar(value=False),
                'wl_var':   tk.DoubleVar(value=float(p['wavelength'])),
                'fwhm_var': tk.DoubleVar(value=float(p['fwhm'])),
            }
            for p in self._preset_data
        }
        self._custom_markers: list[dict]      = []
        self._marker_win:     tk.Toplevel | None = None

        # ── Band parameter results ────────────────────────────────────────────
        self._bp_features:          list[dict]       | None = None
        self._bp_results:           dict             | None = None
        self._bp_unit:              str              | None = None
        self._bp_sources:           dict[str, str]   | None = None
        self._btn_export_params:    ttk.Button       | None = None
        self._btn_visualize_params: ttk.Button       | None = None

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value='')

        # ── Group display mode dropdowns ──────────────────────────────────────
        self._data_group_mode_var = tk.StringVar(value=_GROUP_MODE_VALUES[0])
        self._lib_group_mode_var  = tk.StringVar(value=_GROUP_MODE_VALUES[0])

        # ── Color scheme selection (per side) ────────────────────────────────
        self._data_color_scheme_var = tk.StringVar(value='Dark')
        self._lib_color_scheme_var  = tk.StringVar(value='Dark')

        # ── Color + group-name counters ───────────────────────────────────────
        self._data_color_idx:   int = 0
        self._lib_color_idx:    int = 0
        self._data_group_count: int = 0
        self._lib_group_count:  int = 0

        # ── Filter state ──────────────────────────────────────────────────────
        # Populated by _update_filter_fields() when a library is loaded.
        # Maps display label → field name in the library entry dicts.
        self._lib_filter_fields: dict[str, str] = {}

        self._filter_key_vars:   list[tk.StringVar] = []
        self._filter_val_vars:   list[tk.StringVar] = []
        self._filter_key_combos: list[ttk.Combobox] = []
        self._filter_val_combos: list[ttk.Combobox] = []

        for i in range(_N_FILTER_ROWS):
            self._filter_key_vars.append(tk.StringVar(value=''))
            self._filter_val_vars.append(tk.StringVar(value=_ALL))

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_top_frame()
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=(2, 0))
        self._build_status_bar()
        self._build_main_frame()

    # ── Top frame ────────────────────────────────────────────────────────────

    def _build_top_frame(self) -> None:
        top = ttk.Frame(self, padding=(8, 5, 8, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        # ── Right side first (pack order matters) ─────────────────────────────
        self._btn_build_library = ttk.Button(
            top, text='Build Library', command=self._on_build_library)
        self._btn_build_library.pack(side=tk.RIGHT, padx=(4, 0))

        self._btn_load_library = ttk.Button(
            top, text='Load Library', command=self._on_load_library)
        self._btn_load_library.pack(side=tk.RIGHT, padx=(0, 4))

        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=2)

        # ── Left side ─────────────────────────────────────────────────────────
        self._btn_load_data = ttk.Button(
            top, text='Load Data', command=self._on_load_data)
        self._btn_load_data.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._btn_smooth = ttk.Button(
            top, text='Smooth', command=self._on_smooth)
        self._btn_smooth.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_remove_cr = ttk.Button(
            top, text='Remove Continuum', command=self._on_remove_continuum)
        self._btn_remove_cr.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(top, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._btn_identify_bands = ttk.Button(
            top, text='Identify Bands', command=self._on_identify_bands)
        self._btn_identify_bands.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_band_params = ttk.Button(
            top, text='Band Parameters', command=self._on_band_parameters)
        self._btn_band_params.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_export_params = ttk.Button(
            top, text='Export… ', command=self._on_export_band_params,
            state='disabled')
        self._btn_export_params.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_visualize_params = ttk.Button(
            top, text='Visualize…', command=self._on_visualize_band_params,
            state='disabled')
        self._btn_visualize_params.pack(side=tk.LEFT, padx=(0, 4))

    # ── Main frame: three-column PanedWindow ──────────────────────────────────

    def _build_main_frame(self) -> None:
        main = ttk.Frame(self, padding=(6, 4))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self._build_data_panel(paned)
        self._build_plot_panel(paned)
        self._build_library_panel(paned)

    # ── Status bar ───────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(side=tk.BOTTOM, fill=tk.X, padx=6)
        bar = ttk.Frame(self, padding=(8, 2, 8, 3))
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bar, textvariable=self._status_var, anchor='w',
                  foreground='#555555').pack(side=tk.LEFT, fill=tk.X)

    def _set_status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.update_idletasks()

    # ── Left: data panel ─────────────────────────────────────────────────────

    def _build_data_panel(self, parent: ttk.PanedWindow) -> None:
        left = ttk.LabelFrame(parent, text='Data', padding=(4, 4))
        parent.add(left, weight=2)

        # Row 1: action buttons + group mode dropdown
        btn_frame = ttk.Frame(left)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 1))

        ttk.Button(
            btn_frame, text='Add', command=self._on_data_add_single,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            btn_frame, text='Add Group', command=self._on_data_add_group,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            btn_frame,
            textvariable=self._data_group_mode_var,
            values=_GROUP_MODE_VALUES,
            state='readonly',
            width=14,
        ).pack(side=tk.LEFT)

        # Row 2: offset spinbox
        off_frame = ttk.Frame(left)
        off_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(1, 2))

        ttk.Label(off_frame, text='Offset:').pack(side=tk.LEFT, padx=(0, 4))
        self._data_offset_sb = ttk.Spinbox(
            off_frame,
            textvariable=self._data_offset_var,
            from_=0.0, to=9.90, increment=0.05,
            width=5, format='%.2f',
            command=self._refresh_plot,
        )
        self._data_offset_sb.pack(side=tk.LEFT)
        self._data_offset_sb.bind('<Return>', lambda _e: self._refresh_plot())

        ttk.Label(off_frame, text='Colors:').pack(side=tk.LEFT, padx=(10, 4))
        self._data_color_cb = ttk.Combobox(
            off_frame,
            textvariable=self._data_color_scheme_var,
            values=list(_COLOR_SCHEMES.keys()),
            state='readonly',
            width=10,
        )
        self._data_color_cb.pack(side=tk.LEFT)
        self._data_color_cb.bind(
            '<<ComboboxSelected>>',
            lambda _e: self._on_data_color_scheme_changed(),
        )

        # Available listbox
        avail_lf = ttk.LabelFrame(left, text='Available', padding=4)
        avail_lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=(0, 2))

        self._n_data_label = ttk.Label(avail_lf, text='0 spectra')
        self._n_data_label.pack(anchor=tk.W)

        lb_frame = ttk.Frame(avail_lf)
        lb_frame.pack(fill=tk.BOTH, expand=True)

        vs = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        hs = ttk.Scrollbar(lb_frame, orient=tk.HORIZONTAL)
        self._data_lb = tk.Listbox(
            lb_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=vs.set,
            xscrollcommand=hs.set,
            activestyle='dotbox',
            font=('TkFixedFont', 11),
        )
        vs.config(command=self._data_lb.yview)
        hs.config(command=self._data_lb.xview)
        self._data_lb.grid(row=0, column=0, sticky=tk.NSEW)
        vs.grid(row=0, column=1, sticky=tk.NS)
        hs.grid(row=1, column=0, sticky=tk.EW)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)

        self._data_lb.bind('<<ListboxSelect>>', self._on_data_select)

        # Pool action buttons (under Available)
        pool_rm_frame = ttk.Frame(left)
        pool_rm_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 2))
        ttk.Button(
            pool_rm_frame, text='Remove Selected',
            command=self._on_pool_remove_selected,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            pool_rm_frame, text='Clear All',
            command=self._on_pool_clear_all,
        ).pack(side=tk.LEFT)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=(6, 4))

        # Plotted listbox (small)
        plotted_lf = ttk.LabelFrame(left, text='Plotted', padding=4)
        plotted_lf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 0))

        plb_frame = ttk.Frame(plotted_lf)
        plb_frame.pack(fill=tk.BOTH, expand=True)

        pvs = ttk.Scrollbar(plb_frame, orient=tk.VERTICAL)
        self._data_plotted_lb = tk.Listbox(
            plb_frame,
            selectmode=tk.EXTENDED,
            height=8,
            yscrollcommand=pvs.set,
            activestyle='dotbox',
            font=('TkFixedFont', 11),
        )
        pvs.config(command=self._data_plotted_lb.yview)
        self._data_plotted_lb.grid(row=0, column=0, sticky=tk.NSEW)
        pvs.grid(row=0, column=1, sticky=tk.NS)
        plb_frame.rowconfigure(0, weight=1)
        plb_frame.columnconfigure(0, weight=1)

        # Remove / Clear buttons
        rm_frame = ttk.Frame(left)
        rm_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))

        ttk.Button(
            rm_frame, text='Remove Selected', command=self._on_data_remove_selected,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            rm_frame, text='Clear All', command=self._on_data_clear_all,
        ).pack(side=tk.LEFT)

    # ── Centre: spectral plot ─────────────────────────────────────────────────

    def _build_plot_panel(self, parent: ttk.PanedWindow) -> None:
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=3)

        # Options row
        ctrl_frame = ttk.Frame(plot_frame, padding=(4, 4, 4, 0))
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(ctrl_frame, text='Section gap:').pack(side=tk.LEFT, padx=(0, 4))
        self._section_gap_sb = ttk.Spinbox(
            ctrl_frame,
            textvariable=self._section_gap_var,
            from_=0.0, to=9.90, increment=0.05,
            width=5, format='%.2f',
            command=self._refresh_plot,
        )
        self._section_gap_sb.pack(side=tk.LEFT)
        self._section_gap_sb.bind('<Return>', lambda _e: self._refresh_plot())

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=2)

        ttk.Checkbutton(
            ctrl_frame,
            text='Show labels',
            variable=self._show_labels_var,
            command=self._refresh_plot,
        ).pack(side=tk.LEFT)

        self._cb_show_cr = ttk.Checkbutton(
            ctrl_frame,
            text='Continuum removed',
            variable=self._show_cr_var,
            command=self._refresh_plot,
            state='disabled',
        )
        self._cb_show_cr.pack(side=tk.LEFT, padx=(10, 0))

        self._cb_show_smoothed = ttk.Checkbutton(
            ctrl_frame,
            text='Smoothed',
            variable=self._show_smoothed_var,
            command=self._refresh_plot,
            state='disabled',
        )
        self._cb_show_smoothed.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(ctrl_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=12, pady=2)

        ttk.Button(
            ctrl_frame, text='Markers',
            command=self._on_open_marker_win,
        ).pack(side=tk.LEFT)

        ttk.Separator(plot_frame, orient=tk.HORIZONTAL).pack(
            side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
        self._fig = Figure(figsize=(7, 9), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(top=0.95, bottom=0.08, left=0.10, right=0.97)

        self._ax.set_xlabel('Wavelength (nm)')
        self._ax.set_ylabel('Reflectance (offset for clarity)')
        self._ax.yaxis.set_tick_params(labelleft=True)

        # ── X-axis bottom row ─────────────────────────────────────────────────
        # Pack BEFORE toolbar so it ends up below the toolbar (BOTTOM stacks upward).
        x_row = ttk.Frame(plot_frame, padding=(4, 2, 4, 4))
        x_row.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Label(x_row, text='X-axis:').pack(side=tk.LEFT, padx=(0, 4))
        _unit_cb = ttk.Combobox(
            x_row, textvariable=self._xaxis_unit_var,
            values=['nm', 'µm'], state='readonly', width=4,
        )
        _unit_cb.pack(side=tk.LEFT)
        _unit_cb.bind('<<ComboboxSelected>>', lambda _e: self._on_xaxis_unit_changed())

        ttk.Separator(x_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        ttk.Label(x_row, text='Range:').pack(side=tk.LEFT, padx=(0, 4))
        self._xaxis_min_sb = ttk.Spinbox(
            x_row, textvariable=self._xaxis_min_var,
            from_=100, to=3000, increment=10, width=7, format='%.1f',
            command=self._refresh_plot,
        )
        self._xaxis_min_sb.pack(side=tk.LEFT)
        self._xaxis_min_sb.bind('<Return>', lambda _e: self._refresh_plot())

        ttk.Label(x_row, text='–').pack(side=tk.LEFT, padx=6)

        self._xaxis_max_sb = ttk.Spinbox(
            x_row, textvariable=self._xaxis_max_var,
            from_=100, to=3000, increment=10, width=7, format='%.1f',
            command=self._refresh_plot,
        )
        self._xaxis_max_sb.pack(side=tk.LEFT)
        self._xaxis_max_sb.bind('<Return>', lambda _e: self._refresh_plot())

        self._xaxis_unit_label = ttk.Label(x_row, text='nm')
        self._xaxis_unit_label.pack(side=tk.LEFT, padx=(6, 0))

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        NavigationToolbar2Tk(self._canvas, plot_frame)   # packs at BOTTOM above x_row
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._canvas.draw()

    # ── Right: library panel ──────────────────────────────────────────────────

    def _build_library_panel(self, parent: ttk.PanedWindow) -> None:
        right = ttk.LabelFrame(parent, text='Library', padding=(4, 4))
        parent.add(right, weight=2)

        # Row 1: action buttons + group mode dropdown
        btn_frame = ttk.Frame(right)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 1))

        ttk.Button(
            btn_frame, text='Add', command=self._on_lib_add_single,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            btn_frame, text='Add as Group', command=self._on_lib_add_group,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(
            btn_frame,
            textvariable=self._lib_group_mode_var,
            values=_GROUP_MODE_VALUES,
            state='readonly',
            width=14,
        ).pack(side=tk.LEFT)

        # Row 2: offset spinbox + color scheme
        off_frame = ttk.Frame(right)
        off_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(1, 2))

        ttk.Label(off_frame, text='Offset:').pack(side=tk.LEFT, padx=(0, 4))
        self._lib_offset_sb = ttk.Spinbox(
            off_frame,
            textvariable=self._lib_offset_var,
            from_=0.0, to=9.90, increment=0.05,
            width=5, format='%.2f',
            command=self._refresh_plot,
        )
        self._lib_offset_sb.pack(side=tk.LEFT)
        self._lib_offset_sb.bind('<Return>', lambda _e: self._refresh_plot())

        ttk.Label(off_frame, text='Colors:').pack(side=tk.LEFT, padx=(10, 4))
        self._lib_color_cb = ttk.Combobox(
            off_frame,
            textvariable=self._lib_color_scheme_var,
            values=list(_COLOR_SCHEMES.keys()),
            state='readonly',
            width=10,
        )
        self._lib_color_cb.pack(side=tk.LEFT)
        self._lib_color_cb.bind(
            '<<ComboboxSelected>>',
            lambda _e: self._on_lib_color_scheme_changed(),
        )

        self._build_filter_section(right)
        self._build_library_listbox(right)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=4, pady=(6, 4))

        # Plotted listbox (small)
        lib_plotted_lf = ttk.LabelFrame(right, text='Plotted', padding=4)
        lib_plotted_lf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(0, 0))

        lib_plb_frame = ttk.Frame(lib_plotted_lf)
        lib_plb_frame.pack(fill=tk.BOTH, expand=True)

        lib_pvs = ttk.Scrollbar(lib_plb_frame, orient=tk.VERTICAL)
        self._lib_plotted_lb = tk.Listbox(
            lib_plb_frame,
            selectmode=tk.EXTENDED,
            height=8,
            yscrollcommand=lib_pvs.set,
            activestyle='dotbox',
            font=('TkFixedFont', 11),
        )
        lib_pvs.config(command=self._lib_plotted_lb.yview)
        self._lib_plotted_lb.grid(row=0, column=0, sticky=tk.NSEW)
        lib_pvs.grid(row=0, column=1, sticky=tk.NS)
        lib_plb_frame.rowconfigure(0, weight=1)
        lib_plb_frame.columnconfigure(0, weight=1)

        # Remove / Clear buttons
        lib_rm_frame = ttk.Frame(right)
        lib_rm_frame.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 4))

        ttk.Button(
            lib_rm_frame, text='Remove Selected', command=self._on_lib_remove_selected,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            lib_rm_frame, text='Clear All', command=self._on_lib_clear_all,
        ).pack(side=tk.LEFT)

    def _build_filter_section(self, parent: tk.Widget) -> None:
        filter_lf = ttk.LabelFrame(parent, text='Filters', padding=4)
        filter_lf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))

        for i in range(_N_FILTER_ROWS):
            row_frame = ttk.Frame(filter_lf)
            row_frame.pack(fill=tk.X, pady=1)

            ttk.Label(row_frame, text='Filter by:').pack(side=tk.LEFT, padx=(0, 4))

            key_cb = ttk.Combobox(
                row_frame,
                textvariable=self._filter_key_vars[i],
                values=[],          # populated by _update_filter_fields on load
                state='disabled',
                width=12,
            )
            key_cb.pack(side=tk.LEFT, padx=(0, 4))
            self._filter_key_combos.append(key_cb)

            val_cb = ttk.Combobox(
                row_frame,
                textvariable=self._filter_val_vars[i],
                values=[_ALL],
                state='disabled',
                width=14,
            )
            val_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._filter_val_combos.append(val_cb)

            key_cb.bind(
                '<<ComboboxSelected>>',
                lambda _e, idx=i: self._on_filter_key_changed(idx),
            )
            val_cb.bind('<<ComboboxSelected>>', lambda _e: self._apply_filters())

    def _build_library_listbox(self, parent: tk.Widget) -> None:
        lf = ttk.LabelFrame(parent, text='Available', padding=4)
        lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=2)

        self._n_lib_label = ttk.Label(lf, text='0 spectra')
        self._n_lib_label.pack(anchor=tk.W)

        lb_frame = ttk.Frame(lf)
        lb_frame.pack(fill=tk.BOTH, expand=True)

        vs = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        hs = ttk.Scrollbar(lb_frame, orient=tk.HORIZONTAL)
        self._lib_lb = tk.Listbox(
            lb_frame,
            selectmode=tk.EXTENDED,
            yscrollcommand=vs.set,
            xscrollcommand=hs.set,
            activestyle='dotbox',
            font=('TkFixedFont', 11),
        )
        vs.config(command=self._lib_lb.yview)
        hs.config(command=self._lib_lb.xview)
        self._lib_lb.grid(row=0, column=0, sticky=tk.NSEW)
        vs.grid(row=0, column=1, sticky=tk.NS)
        hs.grid(row=1, column=0, sticky=tk.EW)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)

        self._lib_lb.bind('<<ListboxSelect>>', self._on_lib_select)

    # -----------------------------------------------------------------------
    # Color helper
    # -----------------------------------------------------------------------

    def _next_data_color(self) -> tuple:
        """Return the next color from the active data color scheme."""
        scheme = _COLOR_SCHEMES[self._data_color_scheme_var.get()]
        color  = scheme[self._data_color_idx % len(scheme)]
        self._data_color_idx += 1
        return color

    def _next_lib_color(self) -> tuple:
        """Return the next color from the active library color scheme."""
        scheme = _COLOR_SCHEMES[self._lib_color_scheme_var.get()]
        color  = scheme[self._lib_color_idx % len(scheme)]
        self._lib_color_idx += 1
        return color

    def _on_data_color_scheme_changed(self) -> None:
        """Reassign colors to all data plot items from the new scheme and redraw."""
        self._data_color_idx = 0
        scheme = _COLOR_SCHEMES[self._data_color_scheme_var.get()]
        for i, entry in enumerate(self._data_plot_items):
            entry['color'] = scheme[i % len(scheme)]
        self._refresh_plot()

    def _on_lib_color_scheme_changed(self) -> None:
        """Reassign colors to all library plot items from the new scheme and redraw."""
        self._lib_color_idx = 0
        scheme = _COLOR_SCHEMES[self._lib_color_scheme_var.get()]
        for i, entry in enumerate(self._lib_plot_items):
            entry['color'] = scheme[i % len(scheme)]
        self._refresh_plot()

    # -----------------------------------------------------------------------
    # Feature markers popup
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_preset_features() -> list[dict]:
        """Load preset feature definitions from the YAML file in reference_data."""
        yaml_path = Path(__file__).resolve().parent / 'reference_data' / 'vswir_features.yaml'
        try:
            with open(yaml_path) as f:
                return yaml.safe_load(f).get('features', [])
        except Exception:
            return []

    def _on_open_marker_win(self) -> None:
        """Open the Feature Markers popup, or raise it if already open."""
        if self._marker_win is not None and self._marker_win.winfo_exists():
            self._marker_win.lift()
            return
        self._build_marker_win()

    def _build_marker_win(self) -> None:
        """Create the Feature Markers Toplevel window."""
        win = tk.Toplevel(self)
        win.title('Feature Markers')
        win.geometry('480x560')
        win.resizable(True, True)
        win.wm_attributes('-topmost', True)
        self._marker_win = win

        # ── Preset features section ───────────────────────────────────────────
        preset_lf = ttk.LabelFrame(win, text='Preset Features', padding=(6, 4))
        preset_lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(6, 2))

        # Fixed column header above scroll area
        hdr_row = ttk.Frame(preset_lf)
        hdr_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(hdr_row, text='Feature', width=26).pack(side=tk.LEFT)
        ttk.Label(hdr_row, text='Centre (nm)', width=10, anchor='e').pack(side=tk.LEFT)
        ttk.Label(hdr_row, text='FWHM (nm)',   width=10, anchor='e').pack(side=tk.LEFT)
        ttk.Separator(preset_lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 2))

        # Scrollable preset rows
        p_outer = ttk.Frame(preset_lf)
        p_outer.pack(fill=tk.BOTH, expand=True)
        p_canvas = tk.Canvas(p_outer, borderwidth=0, highlightthickness=0)
        p_vsb = ttk.Scrollbar(p_outer, orient=tk.VERTICAL, command=p_canvas.yview)
        p_canvas.configure(yscrollcommand=p_vsb.set)
        p_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        p_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        p_rows_frame = ttk.Frame(p_canvas)
        _p_cwin = p_canvas.create_window((0, 0), window=p_rows_frame, anchor='nw')
        p_rows_frame.bind(
            '<Configure>',
            lambda _e: p_canvas.configure(scrollregion=p_canvas.bbox('all')),
        )
        p_canvas.bind(
            '<Configure>',
            lambda e: p_canvas.itemconfigure(_p_cwin, width=e.width),
        )

        current_group: str | None = None
        for preset in self._preset_data:
            name  = preset['name']
            group = preset.get('group', '')
            pvars = self._preset_vars[name]

            if group != current_group:
                current_group = group
                g_row = ttk.Frame(p_rows_frame)
                g_row.pack(fill=tk.X, pady=(6, 0))
                ttk.Label(
                    g_row, text=group,
                    font=('TkDefaultFont', 9, 'bold'),
                ).pack(side=tk.LEFT)
                ttk.Separator(p_rows_frame, orient=tk.HORIZONTAL).pack(
                    fill=tk.X, pady=(1, 2))

            row = ttk.Frame(p_rows_frame)
            row.pack(fill=tk.X, pady=1)
            ttk.Checkbutton(
                row, text=name, variable=pvars['enabled'], width=26,
                command=self._refresh_plot,
            ).pack(side=tk.LEFT)
            for var, lo, hi in [
                (pvars['wl_var'],   100.0, 3000.0),
                (pvars['fwhm_var'],   1.0,  500.0),
            ]:
                sb = ttk.Spinbox(row, textvariable=var, from_=lo, to=hi,
                                 increment=1.0, width=8, format='%.1f',
                                 command=self._refresh_plot)
                sb.pack(side=tk.LEFT, padx=(4, 0))
                sb.bind('<Return>',   lambda _e: self._refresh_plot())
                sb.bind('<FocusOut>', lambda _e: self._refresh_plot())

        # ── Custom markers section ────────────────────────────────────────────
        custom_lf = ttk.LabelFrame(win, text='Custom Markers', padding=(6, 4))
        custom_lf.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(2, 6))

        # Add button + column labels
        top_bar = ttk.Frame(custom_lf)
        top_bar.pack(fill=tk.X, pady=(0, 2))
        ttk.Button(top_bar, text='Add Marker', command=self._add_marker).pack(side=tk.LEFT)

        col_hdr = ttk.Frame(custom_lf)
        col_hdr.pack(fill=tk.X, pady=(2, 2))
        ttk.Label(col_hdr, text='Name',           width=15).pack(side=tk.LEFT)
        ttk.Label(col_hdr, text='Centre (nm)',     width=11).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(col_hdr, text='FWHM (nm)',       width=9 ).pack(side=tk.LEFT, padx=(4, 0))

        ttk.Separator(custom_lf, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 4))

        # Scrollable rows
        outer = ttk.Frame(custom_lf)
        outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0, height=80)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._marker_rows_frame = ttk.Frame(canvas)
        _cwin = canvas.create_window((0, 0), window=self._marker_rows_frame, anchor='nw')

        self._marker_rows_frame.bind(
            '<Configure>',
            lambda _e: canvas.configure(scrollregion=canvas.bbox('all')),
        )
        canvas.bind(
            '<Configure>',
            lambda e: canvas.itemconfigure(_cwin, width=e.width),
        )

        # Rebuild rows for any pre-existing custom markers
        for marker in self._custom_markers:
            self._add_marker_row(marker)

    def _add_marker(self) -> None:
        """Append a new custom marker with default name, wavelength and FWHM."""
        n = len(self._custom_markers) + 1
        marker: dict = {
            'name_var': tk.StringVar(value=f'Feature {n:02d}'),
            'wl_var':   tk.DoubleVar(value=1000.0),
            'fwhm_var': tk.DoubleVar(value=30.0),
            'frame':    None,
        }
        self._custom_markers.append(marker)
        if self._marker_win is not None and self._marker_win.winfo_exists():
            self._add_marker_row(marker)
        self._refresh_plot()

    def _add_marker_row(self, marker: dict) -> None:
        """
        Create the widget row inside the custom section for *marker*.

        Parameters
        ----------
        marker : dict
            Keys: ``name_var``, ``wl_var``, ``fwhm_var``, ``frame`` (set here).
        """
        row = ttk.Frame(self._marker_rows_frame, padding=(2, 1))
        row.pack(fill=tk.X, pady=1)
        marker['frame'] = row

        name_e = ttk.Entry(row, textvariable=marker['name_var'], width=15)
        name_e.pack(side=tk.LEFT, padx=(0, 4))
        name_e.bind('<Return>',   lambda _e: self._refresh_plot())
        name_e.bind('<FocusOut>', lambda _e: self._refresh_plot())

        for var, lo, hi in [
            (marker['wl_var'],   100.0, 3000.0),
            (marker['fwhm_var'],   1.0,  500.0),
        ]:
            sb = ttk.Spinbox(
                row, textvariable=var,
                from_=lo, to=hi, increment=1.0,
                width=8, format='%.1f',
                command=self._refresh_plot,
            )
            sb.pack(side=tk.LEFT, padx=(0, 4))
            sb.bind('<Return>',   lambda _e: self._refresh_plot())
            sb.bind('<FocusOut>', lambda _e: self._refresh_plot())

        ttk.Button(
            row, text='×', width=2,
            command=lambda m=marker: self._remove_marker(m),
        ).pack(side=tk.LEFT)

    def _remove_marker(self, marker: dict) -> None:
        """Remove *marker* from the custom list, destroy its row, and redraw."""
        self._custom_markers.remove(marker)
        if marker['frame'] is not None:
            marker['frame'].destroy()
        self._refresh_plot()

    # -----------------------------------------------------------------------
    # Plotted listbox sync
    # -----------------------------------------------------------------------

    def _sync_data_plotted_lb(self) -> None:
        self._data_plotted_lb.delete(0, tk.END)
        for entry in self._data_plot_items:
            self._data_plotted_lb.insert(tk.END, entry['name'])

    def _sync_lib_plotted_lb(self) -> None:
        self._lib_plotted_lb.delete(0, tk.END)
        for entry in self._lib_plot_items:
            self._lib_plotted_lb.insert(tk.END, entry['name'])

    # -----------------------------------------------------------------------
    # Callbacks — plotted remove / clear
    # -----------------------------------------------------------------------

    def _on_pool_remove_selected(self) -> None:
        """Remove selected entries from the data pool (Available listbox)."""
        for idx in sorted(self._data_lb.curselection(), reverse=True):
            del self._data_pool[idx]
            self._data_lb.delete(idx)
        self._n_data_label.config(text=f'{len(self._data_pool)} spectra')

    def _on_pool_clear_all(self) -> None:
        """Remove all entries from the data pool (Available listbox)."""
        self._data_pool.clear()
        self._data_lb.delete(0, tk.END)
        self._n_data_label.config(text='0 spectra')

    def _on_data_remove_selected(self) -> None:
        """Remove selected entries from the data plot list and redraw."""
        for idx in sorted(self._data_plotted_lb.curselection(), reverse=True):
            del self._data_plot_items[idx]
        self._sync_data_plotted_lb()
        self._refresh_plot()

    def _on_data_clear_all(self) -> None:
        """Remove all data plot entries and redraw."""
        self._data_plot_items.clear()
        self._sync_data_plotted_lb()
        self._refresh_plot()

    def _on_lib_remove_selected(self) -> None:
        """Remove selected entries from the library plot list and redraw."""
        for idx in sorted(self._lib_plotted_lb.curselection(), reverse=True):
            del self._lib_plot_items[idx]
        self._sync_lib_plotted_lb()
        self._refresh_plot()

    def _on_lib_clear_all(self) -> None:
        """Remove all library plot entries and redraw."""
        self._lib_plot_items.clear()
        self._sync_lib_plotted_lb()
        self._refresh_plot()

    # -----------------------------------------------------------------------
    # Filter helpers
    # -----------------------------------------------------------------------

    def _update_filter_fields(self) -> None:
        """
        Discover filterable metadata fields from the loaded library and update
        the filter keyword dropdowns.

        A field is included when:
        - it is not in ``_LIB_NON_FILTER_FIELDS`` (not data/xaxis/display-name),
        - all its values are scalar strings or empty,
        - it has between 2 and 200 distinct non-empty values (enough variety to
          be useful, not so many that the value dropdown becomes unwieldy).

        Results are sorted from fewest to most unique values (coarse categories
        first), which puts the most useful high-level filters at the top of the
        list.
        """
        # Tally unique non-empty string values per field
        field_vals: dict[str, set[str]] = {}
        for entry in self._library.values():
            for key, val in entry.items():
                if key in _LIB_NON_FILTER_FIELDS:
                    continue
                if not isinstance(val, str):
                    continue
                val = val.strip()
                if val:
                    field_vals.setdefault(key, set()).add(val)

        # Keep only fields with 2 ≤ n_unique ≤ 200
        eligible = {
            k: v for k, v in field_vals.items()
            if 2 <= len(v) <= 200
        }
        # Sort fewest-unique first (coarsest categories at the top)
        sorted_fields = sorted(eligible, key=lambda k: len(eligible[k]))

        # Build display-name → field-name mapping
        self._lib_filter_fields = {
            _LIB_FIELD_DISPLAY.get(f, f.replace('_', ' ').title()): f
            for f in sorted_fields
        }

        display_labels = list(self._lib_filter_fields.keys())

        # Update key combos with discovered fields; reset val combos
        for i, key_cb in enumerate(self._filter_key_combos):
            key_cb.config(state='readonly', values=display_labels)
            # Pre-select the i-th field if available, otherwise clear
            default = display_labels[i] if i < len(display_labels) else ''
            self._filter_key_vars[i].set(default)
            self._filter_val_vars[i].set(_ALL)
            self._filter_val_combos[i].config(state='disabled', values=[_ALL])

        self._populate_all_val_combos()

    def _on_filter_key_changed(self, row: int) -> None:
        """Repopulate the value dropdown when the keyword selector changes."""
        self._filter_val_vars[row].set(_ALL)
        self._populate_val_combo(row)
        self._apply_filters()

    def _populate_val_combo(self, row: int) -> None:
        """Fill the value Combobox for *row* with unique values from the library."""
        field = self._lib_filter_fields.get(self._filter_key_vars[row].get(), '')
        cb    = self._filter_val_combos[row]

        if not self._library or not field:
            cb.config(state='disabled', values=[_ALL])
            self._filter_val_vars[row].set(_ALL)
            return

        seen = set()
        for entry in self._library.values():
            v = entry.get(field)
            if v and str(v).strip():
                seen.add(str(v).strip())

        values = [_ALL] + sorted(seen)
        cb.config(state='readonly', values=values)
        if self._filter_val_vars[row].get() not in values:
            self._filter_val_vars[row].set(_ALL)

    def _populate_all_val_combos(self) -> None:
        for i in range(_N_FILTER_ROWS):
            self._populate_val_combo(i)

    def _apply_filters(self) -> None:
        """Filter the library and refresh the listbox."""
        filtered = dict(self._library)
        for i in range(_N_FILTER_ROWS):
            sel   = self._filter_val_vars[i].get()
            field = self._lib_filter_fields.get(self._filter_key_vars[i].get(), '')
            if sel != _ALL and field:
                filtered = {
                    sid: v for sid, v in filtered.items()
                    if str(v.get(field, '')).strip() == sel
                }

        self._filtered_ids = list(filtered.keys())
        self._lib_lb.delete(0, tk.END)
        for sid in self._filtered_ids:
            entry = filtered[sid]
            label = entry.get('label') or entry.get('sample_name') or entry.get('name') or str(sid)
            self._lib_lb.insert(tk.END, label)

        self._n_lib_label.config(text=f'{len(self._filtered_ids)} spectra')

    # -----------------------------------------------------------------------
    # Plot helpers
    # -----------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        """
        Redraw the spectral canvas with vertically offset plot entries.

        Library entries occupy the bottom (index 0 = lowest y); data entries
        occupy the top.  Two separate legends are drawn — one per side.
        Y-axis tick labels are suppressed; absolute values carry no meaning.
        """
        self._ax.cla()
        unit      = self._xaxis_unit_var.get()
        wl_scale  = 1e-3 if unit == 'µm' else 1.0
        use_cr = self._show_cr_var.get() and self._cr_available
        self._ax.set_xlabel(f'Wavelength ({unit})')
        self._ax.set_ylabel('CR Reflectance (offset)' if use_cr else 'Reflectance (offset)')
        self._ax.yaxis.set_tick_params(labelleft=True)

        data_offset  = self._data_offset_var.get()
        lib_offset   = self._lib_offset_var.get()
        section_gap  = self._section_gap_var.get()
        n_lib        = len(self._lib_plot_items)

        for i, entry in enumerate(self._lib_plot_items):
            self._plot_entry(entry, y_shift=i * lib_offset, wl_scale=wl_scale, use_cr=use_cr)

        data_base = (n_lib * lib_offset + section_gap) if n_lib > 0 else 0.0
        for i, entry in enumerate(self._data_plot_items):
            self._plot_entry(entry, y_shift=data_base + data_offset * i,
                             wl_scale=wl_scale, use_cr=use_cr)

        if self._show_labels_var.get():
            for i, entry in enumerate(self._lib_plot_items):
                self._draw_label(entry, y_shift=i * lib_offset, side='right',
                                 wl_scale=wl_scale, use_cr=use_cr)
            for i, entry in enumerate(self._data_plot_items):
                self._draw_label(entry, y_shift=data_base + data_offset * i, side='left',
                                 wl_scale=wl_scale, use_cr=use_cr)

        # Feature markers — presets (checked) and custom entries; wl/fwhm stored in nm
        all_markers = []
        for p in self._preset_data:
            pvars = self._preset_vars[p['name']]
            if pvars['enabled'].get():
                all_markers.append({
                    'name': p['name'],
                    'wl':   pvars['wl_var'].get() * wl_scale,
                    'fwhm': pvars['fwhm_var'].get() * wl_scale,
                })
        for m in self._custom_markers:
            all_markers.append({
                'name': m['name_var'].get(),
                'wl':   m['wl_var'].get() * wl_scale,
                'fwhm': m['fwhm_var'].get() * wl_scale,
            })
        if all_markers:
            trans = mtransforms.blended_transform_factory(
                self._ax.transData, self._ax.transAxes)
            for marker in all_markers:
                wl   = marker['wl']
                fwhm = marker['fwhm']
                name = marker['name']
                self._ax.axvspan(
                    wl - fwhm / 2, wl + fwhm / 2,
                    alpha=0.12, color='dimgray', zorder=0, lw=0,
                )
                self._ax.axvline(wl, color='dimgray', lw=0.9, ls='--', alpha=0.6, zorder=1)
                if name:
                    self._ax.text(
                        wl, 0.97, name, transform=trans,
                        rotation=90, va='top', ha='center',
                        fontsize=8, color='dimgray', clip_on=True, zorder=2,
                    )

        xmin = self._xaxis_min_var.get()
        xmax = self._xaxis_max_var.get()
        if xmin < xmax:
            self._ax.set_xlim(xmin, xmax)

        self._canvas.draw_idle()

    def _draw_label(self, entry: dict, y_shift: float, side: str,
                    wl_scale: float = 1.0, use_cr: bool = False) -> None:
        """
        Draw an inline text label anchored just inside the plot edge for *entry*.

        The text box is positioned so that its bottom-left corner (data side)
        or bottom-right corner (library side) sits a few points away from the
        spectral line endpoint, keeping the label fully within the axes.

        Parameters
        ----------
        entry : dict
            Plot entry dict (same schema as accepted by ``_plot_entry``).
        y_shift : float
            Vertical offset applied to this entry (same value used when plotting).
        side : str
            ``'left'``  — anchored at ``xaxis[0]``,  bottom-left of text box.
            ``'right'`` — anchored at ``xaxis[-1]``, bottom-right of text box.
        wl_scale : float
            Same scale factor used when plotting (nm → display unit).
        use_cr : bool
            When True, use ``cr_data`` for the y-anchor if available.
        """
        xaxis = entry['xaxis'] * wl_scale
        color = entry['color']
        name  = entry['name']
        data  = entry['cr_data'] if (use_cr and 'cr_data' in entry) else self._display_data(entry)

        if side == 'left':
            xi        = 0
            x         = xaxis[0]
            ha        = 'left'
            x_pts_off = -10     # shift text rightward away from the line start
        else:
            xi        = -1
            x         = xaxis[-1]
            ha        = 'right'
            x_pts_off = 10    # shift text leftward away from the line end

        if entry['kind'] == 'single':
            y_edge = float(data[xi])
        elif entry.get('display_mode', 'median') == 'all':
            y_edge = float(np.mean(data[:, xi]))
        else:
            y_edge = float(np.median(data[:, xi]))

        self._ax.annotate(
            name,
            xy=(x, y_edge + y_shift),
            xytext=(x_pts_off, 10),
            textcoords='offset points',
            ha=ha, va='bottom',
            color=color, fontsize=9, clip_on=True,
        )

    def _plot_entry(self, entry: dict, y_shift: float,
                    wl_scale: float = 1.0, use_cr: bool = False) -> None:
        """
        Render one plot entry at *y_shift*.

        Parameters
        ----------
        entry : dict
            Keys: ``kind`` ('single'|'group'), ``name``, ``xaxis``, ``data``,
            ``color``, and for groups ``display_mode`` ('median'|'all').
            May also contain ``cr_data`` with the continuum-removed counterpart.
        y_shift : float
            Vertical offset applied to all plotted values.
        wl_scale : float
            Multiplicative scale applied to the x-axis before plotting (e.g.
            0.001 to convert nm → µm).
        use_cr : bool
            When True, plot ``cr_data`` instead of ``data`` if available.
        """
        xaxis = entry['xaxis'] * wl_scale
        name  = entry['name']
        color = entry['color']
        data  = entry['cr_data'] if (use_cr and 'cr_data' in entry) else self._display_data(entry)

        if entry['kind'] == 'single':
            self._ax.plot(xaxis, data + y_shift, color=color, label=name)
            return

        if entry.get('display_mode', 'median') == 'all':
            for j, row in enumerate(data):
                self._ax.plot(
                    xaxis, row + y_shift, color=color, lw=1.0, label=name if j == 0 else '_',
                )
            return

        # Median ± std
        med = np.median(data, axis=0)
        std = np.std(data,    axis=0)
        self._ax.plot(xaxis, med + y_shift, color=color, lw=1.0, label=name)
        self._ax.fill_between(
            xaxis,
            med - std + y_shift,
            med + std + y_shift,
            lw=0, color=color, alpha=0.25,
        )

    # -----------------------------------------------------------------------
    # Callbacks — top frame
    # -----------------------------------------------------------------------

    def _on_xaxis_unit_changed(self) -> None:
        """
        Convert min/max range spinbox values to the new unit and reconfigure
        the spinboxes (from/to/increment/format) accordingly.
        """
        new_unit = self._xaxis_unit_var.get()
        lo = self._xaxis_min_var.get()
        hi = self._xaxis_max_var.get()
        if new_unit == 'µm':
            self._xaxis_min_var.set(round(lo * 1e-3, 3))
            self._xaxis_max_var.set(round(hi * 1e-3, 3))
            for sb in (self._xaxis_min_sb, self._xaxis_max_sb):
                if sb is not None:
                    sb.config(from_=0.1, to=3.0, increment=0.01, format='%.3f')
        else:
            self._xaxis_min_var.set(round(lo * 1e3, 1))
            self._xaxis_max_var.set(round(hi * 1e3, 1))
            for sb in (self._xaxis_min_sb, self._xaxis_max_sb):
                if sb is not None:
                    sb.config(from_=100, to=3000, increment=10, format='%.1f')
        if self._xaxis_unit_label is not None:
            self._xaxis_unit_label.config(text=new_unit)
        self._refresh_plot()

    # -----------------------------------------------------------------------
    # Display-data helpers (smooth / CR interaction)
    # -----------------------------------------------------------------------

    def _display_data(self, entry: dict) -> np.ndarray:
        """Return the currently active display data for *entry*.

        Returns ``smoothed_data`` when smoothing is active and available,
        otherwise the raw ``data``.
        """
        if self._smooth_available and self._show_smoothed_var.get():
            return entry.get('smoothed_data', entry['data'])
        return entry['data']

    def _smooth_entry(self, item: dict) -> None:
        """Compute and cache ``smoothed_data`` for *item* from ``_smooth_params``."""
        if not self._smooth_params:
            return
        p     = self._smooth_params
        xaxis = item['xaxis']
        raw   = item['data']
        if item['kind'] == 'single':
            item['smoothed_data'] = smooth_spectrum(
                xaxis, raw, p['smooth_method'],
                window_nm=p['smooth_window_nm'], poly_order=p['smooth_polyorder'],
            )
        else:
            item['smoothed_data'] = np.stack([
                smooth_spectrum(xaxis, row, p['smooth_method'],
                                window_nm=p['smooth_window_nm'],
                                poly_order=p['smooth_polyorder'])
                for row in raw
            ])

    def _cr_entry(self, item: dict) -> None:
        """Compute and cache ``cr_data`` for *item* from the current display data."""
        if not self._cr_available:
            return
        item['cr_data'] = remove_continuum(
            item['xaxis'], self._display_data(item), wl_range=self._cr_wl_range,
        )

    # -----------------------------------------------------------------------
    # Callbacks — toolbar
    # -----------------------------------------------------------------------

    def _on_smooth(self) -> None:
        """
        Open the smoothing dialog, apply smoothing to all plotted spectra, and
        enable the "Smoothed" display toggle.

        Parameters are stored in ``_smooth_params`` and auto-applied to any
        spectra added to the plot afterwards.  Continuum-removed data already
        cached is left untouched; to apply CR to smoothed data, re-run
        "Remove Continuum" after smoothing.
        """
        all_items = self._data_plot_items + self._lib_plot_items
        if not all_items:
            messagebox.showinfo('Smooth Spectra',
                                'No spectra are plotted.', parent=self)
            return

        dlg = SmoothingDialog(self)
        if dlg.cancelled:
            return

        n = len(all_items)
        self._set_status(f'Smoothing {n} spectra…')
        self._smooth_params = dlg.params
        for item in all_items:
            self._smooth_entry(item)

        self._smooth_available = True
        if self._cb_show_smoothed is not None:
            self._cb_show_smoothed.config(state='normal')
        self._show_smoothed_var.set(True)
        self._refresh_plot()
        self._set_status(f'Smoothing applied — {n} spectra')

    def _on_remove_continuum(self) -> None:
        """
        Open the continuum-removal dialog, then compute and cache CR data for
        all current plot items.

        The chosen wavelength range is stored in ``_cr_wl_range`` and reused
        automatically when new spectra are added to the plot after CR has been
        run.  Re-pressing the button opens the dialog again so the range can
        be changed; all plot items are recomputed on each run.
        """
        # Infer the default range from the first loaded data xaxis, if any.
        if self._data_pool:
            xaxis       = self._data_pool[0]['xaxis']
            default_rng = (float(xaxis[0]), float(xaxis[-1]))
        else:
            default_rng = (400.0, 2500.0)

        dlg = ContinuumRemovalDialog(
            self,
            default_range=default_rng,
            last_range=self._cr_wl_range,
        )
        if dlg.cancelled:
            return

        all_items = self._data_plot_items + self._lib_plot_items
        if not all_items:
            return

        n = len(all_items)
        self._cr_wl_range = dlg.wl_range
        for i, entry in enumerate(all_items):
            self._set_status(f'Continuum removal — {i + 1}/{n}')
            entry['cr_data'] = remove_continuum(
                entry['xaxis'], self._display_data(entry), wl_range=self._cr_wl_range)
        self._cr_available = True
        if self._cb_show_cr is not None:
            self._cb_show_cr.config(state='normal')
        self._refresh_plot()
        self._set_status(f'Continuum removal applied — {n} spectra')

    def _on_band_parameters(self) -> None:
        """
        Open the band-parameters dialog, compute parameters for each selected
        feature on the chosen set of spectra, and display the results table.

        Groups are represented by their median spectrum.  Preset features that
        are currently active as markers are pre-checked.  Custom markers
        (including unmatched bands identified by the Band Identification
        workflow) appear as a separate group and are pre-checked by default.

        Scope radio buttons in the dialog control whether only plotted spectra
        or all loaded / available spectra are analysed on each side.
        """
        # Enabled preset names
        active_names: set[str] = {
            name for name, pvars in self._preset_vars.items()
            if pvars['enabled'].get()
        }

        # Append custom markers (includes identified-but-unmatched bands) as
        # pseudo-presets; they are always pre-checked.
        custom_as_presets: list[dict] = [
            {
                'name':       m['name_var'].get(),
                'group':      'Custom / Identified',
                'wavelength': m['wl_var'].get(),
                'fwhm':       m['fwhm_var'].get(),
            }
            for m in self._custom_markers
        ]
        all_preset_data = self._preset_data + custom_as_presets
        active_names |= {m['name_var'].get() for m in self._custom_markers}

        dlg = BandParametersDialog(
            self, all_preset_data, active_names,
            n_data_plotted=len(self._data_plot_items),
            n_data_available=len(self._data_pool),
            n_lib_plotted=len(self._lib_plot_items),
            n_lib_available=len(self._filtered_ids),
        )
        if dlg.cancelled:
            return

        # Build item lists based on scope selection
        if dlg.data_scope == 'all':
            data_items: list[dict] = [
                {'kind': 'single', 'name': e['name'], 'xaxis': e['xaxis'], 'data': e['data']}
                for e in self._data_pool
            ]
        else:
            data_items = self._data_plot_items

        if dlg.lib_scope == 'all':
            lib_items: list[dict] = []
            for sid in self._filtered_ids:
                e = self._library[sid]
                lib_items.append({
                    'kind':  'single',
                    'name':  e.get('label') or e.get('sample_name') or e.get('name') or str(sid),
                    'xaxis': e['xaxis'],
                    'data':  e['data'],
                })
        else:
            lib_items = self._lib_plot_items

        all_items = data_items + lib_items
        if not all_items:
            messagebox.showinfo('Band Parameters',
                                'No spectra to analyse.', parent=self)
            return

        sources: dict[str, str] = {}
        for entry in data_items:
            sources[entry['name']] = 'Data'
        for entry in lib_items:
            sources[entry['name']] = 'Library'

        n = len(all_items)
        m = len(dlg.selected)
        results: dict[str, dict] = {}
        for i, entry in enumerate(all_items):
            self._set_status(f'Band parameters — {i + 1}/{n}')
            xaxis    = entry['xaxis']
            # Use _display_data (smooth/CR aware) for plotted items; raw data otherwise.
            raw      = self._display_data(entry) if 'color' in entry else entry['data']
            spectrum = raw if entry['kind'] == 'single' else np.median(raw, axis=0)

            feat_results: dict[str, dict | None] = {}
            for feat in dlg.selected:
                feat_results[feat['name']] = band_parameters(
                    xaxis, spectrum, wl_range=feat['wl_range'])
            results[entry['name']] = feat_results

        self._bp_features = dlg.selected
        self._bp_results  = results
        self._bp_unit     = self._xaxis_unit_var.get()
        self._bp_sources  = sources
        if self._btn_export_params is not None:
            self._btn_export_params.config(state='normal')
        if self._btn_visualize_params is not None:
            self._btn_visualize_params.config(state='normal')
        self._set_status(f'Band parameters done — {n} spectra, {m} feature{"s" if m != 1 else ""}')

    def _on_export_band_params(self) -> None:
        if self._bp_features is None or self._bp_results is None:
            return
        from tkinter import filedialog
        import csv
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            title='Export band parameters',
        )
        if not path:
            return
        unit  = self._bp_unit or 'nm'
        scale = 1e-3 if unit == 'µm' else 1.0
        sources = self._bp_sources or {}
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            header = ['Spectrum', 'Source']
            for feat in self._bp_features:
                n = feat['name']
                for key, tmpl, scaled, _ in _BAND_METRIC_SPEC:
                    lbl = tmpl.format(unit=unit) if scaled else tmpl
                    header.append(f'{n} {lbl.replace(chr(10), " ")}')
            writer.writerow(header)
            for sp_name, feat_results in self._bp_results.items():
                row: list = [sp_name, sources.get(sp_name, '')]
                for feat in self._bp_features:
                    bp = feat_results.get(feat['name'])
                    if bp is None:
                        row += [''] * len(_BAND_METRIC_SPEC)
                    else:
                        for key, _, scaled, _ in _BAND_METRIC_SPEC:
                            val = bp.get(key)
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                row.append('')
                            else:
                                row.append(val * scale if scaled else val)
                writer.writerow(row)
        self._set_status(f'Band parameters saved — {Path(path).name}')

    def _on_visualize_band_params(self) -> None:
        if self._bp_features is None or self._bp_results is None:
            return
        BandVizWindow(self, self._bp_features, self._bp_results,
                      self._bp_unit or 'nm',
                      sources=self._bp_sources or {})

    def _on_identify_bands(self) -> None:
        """
        Open the Band Identification pre-run dialog, run ``detect_bands`` on
        each selected spectrum, merge candidates across spectra, and display
        the results dialog.

        Groups are represented by their median spectrum.  Candidates from
        successive spectra are merged with the accumulated list by proximity
        (using the same ``match_tolerance_nm`` chosen in the dialog): the first
        occurrence's parameters are kept, and the spectrum name is appended to
        ``seen_in``.
        """
        all_items = self._data_plot_items + self._lib_plot_items
        if not all_items:
            messagebox.showinfo('Identify Bands',
                                'No spectra are plotted.', parent=self)
            return

        dlg = BandIdentificationDialog(
            self, all_items,
            n_data_plotted=len(self._data_plot_items),
            n_data_available=len(self._data_pool),
            n_lib_plotted=len(self._lib_plot_items),
            n_lib_available=len(self._filtered_ids),
        )
        if dlg.cancelled:
            return

        params = dlg.params

        # Build the processing list according to scope selections.
        if dlg.data_scope == 'all':
            data_items: list[dict] = [
                {'kind': 'single', 'name': e['name'],
                 'xaxis': e['xaxis'], 'data': e['data']}
                for e in self._data_pool
            ]
        else:
            data_items = [e for e in self._data_plot_items
                          if e['name'] in dlg.selected_names]

        if dlg.lib_scope == 'all':
            lib_items: list[dict] = []
            for sid in self._filtered_ids:
                e = self._library[sid]
                lib_items.append({
                    'kind':  'single',
                    'name':  e.get('label') or e.get('sample_name') or e.get('name') or str(sid),
                    'xaxis': e['xaxis'],
                    'data':  e['data'],
                })
        else:
            lib_items = [e for e in self._lib_plot_items
                         if e['name'] in dlg.selected_names]

        to_process = data_items + lib_items
        n_proc = len(to_process)

        # ── Run detection ─────────────────────────────────────────────────────
        raw_results: list[tuple[str, list[dict]]] = []
        for i, entry in enumerate(to_process):
            self._set_status(f'Detecting bands — {i + 1}/{n_proc}')
            xaxis    = entry['xaxis']
            raw      = self._display_data(entry) if 'color' in entry else entry['data']
            spectrum = raw if entry['kind'] == 'single' else np.median(raw, axis=0)
            candidates = detect_bands(
                xaxis, spectrum, self._preset_data,
                smooth_method=params['smooth_method'],
                smooth_window_nm=params['smooth_window_nm'],
                smooth_polyorder=params['smooth_polyorder'],
                min_prominence=params['min_prominence'],
                min_width_nm=params['min_width_nm'],
                min_depth=params['min_depth'],
                match_tolerance_nm=params['match_tolerance_nm'],
            )
            raw_results.append((entry['name'], candidates))

        # ── Cross-spectrum merge ──────────────────────────────────────────────
        tol    = params['match_tolerance_nm']
        merged: list[dict] = []
        for sp_name, candidates in raw_results:
            for cand in candidates:
                existing = next(
                    (m for m in merged
                     if abs(cand['wl_min'] - m['wl_min']) < tol),
                    None,
                )
                if existing is not None:
                    existing['seen_in'].append(sp_name)
                else:
                    merged.append({**cand, 'seen_in': [sp_name]})
        merged.sort(key=lambda c: c['wl_min'])

        if not merged:
            self._set_status('Band identification — no features detected')
            messagebox.showinfo(
                'Identify Bands',
                'No absorption features detected with the current parameters.',
                parent=self,
            )
            return

        k = len(merged)
        self._set_status(f'Band identification done — {k} feature{"s" if k != 1 else ""} found')
        BandIdentificationResultsDialog(self, merged, on_add=self._apply_identified_markers)

    def _apply_identified_markers(self, candidates: list[dict]) -> None:
        """
        Apply a list of identified candidates to the marker system.

        Candidates with a ``matched_name`` that corresponds to a known preset
        have that preset enabled and its wavelength / FWHM updated to the
        detected values.  Unmatched (or externally named) candidates are added
        as custom markers.
        """
        for cand in candidates:
            name = cand.get('matched_name')
            if name is not None and name in self._preset_vars:
                pvars = self._preset_vars[name]
                pvars['enabled'].set(True)
                pvars['wl_var'].set(round(cand['wl_min'], 1))
                pvars['fwhm_var'].set(round(cand['fwhm'], 1))
            else:
                label = name if name else f"Unknown @ {cand['wl_min']:.0f} nm"
                marker: dict = {
                    'name_var': tk.StringVar(value=label),
                    'wl_var':   tk.DoubleVar(value=round(cand['wl_min'], 1)),
                    'fwhm_var': tk.DoubleVar(value=round(cand['fwhm'], 1)),
                    'frame':    None,
                }
                self._custom_markers.append(marker)
                if self._marker_win is not None and self._marker_win.winfo_exists():
                    self._add_marker_row(marker)
        self._refresh_plot()

    def _on_load_data(self) -> None:
        """Show a format picker, open a file chooser, and append spectra to the data list."""
        dlg = LoadFormatDialog(self)
        if dlg.result is None:
            return

        if dlg.result == 'asd':
            path = filedialog.askopenfilename(
                title='Load ASD data',
                filetypes=[('ASD text files', '*.txt'), ('All files', '*.*')],
            )
            if not path:
                return
            try:
                xaxis, spectra = _read_vswir_asd(Path(path))
            except ValueError as exc:
                messagebox.showerror('Load Data', str(exc), parent=self)
                return
            wr_dlg = WhiteReferenceCorrectionDialog(self, xaxis, spectra, source_path=Path(path))
            if wr_dlg.cancelled:
                return
            source = Path(path)
            for item in wr_dlg.result:
                self._data_pool.append({
                    'name':   item['name'],
                    'xaxis':  item['xaxis'],
                    'data':   item['data'],
                    'source': source,
                })
                self._data_lb.insert(tk.END, item['name'])
            n = len(wr_dlg.result)
            self._n_data_label.config(text=f'{len(self._data_pool)} spectra')
            self._set_status(f'Data loaded — {n} spectra ({source.name})')
        else:
            path = filedialog.askopenfilename(
                title='Load CSV data',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
            )
            if not path:
                return
            self._load_data_from_path(Path(path))

    def _load_data_from_path(self, path: Path) -> None:
        """
        Parse a reflectance CSV and append each spectrum column to ``_data_pool``.

        Populates the left listbox but does not plot anything; use the Add
        buttons to move selected spectra onto the plot.

        Parameters
        ----------
        path : Path
            Path to the CSV file.
        """
        self._set_status(f'Loading {path.name}…')
        try:
            xaxis, spectra = _read_vswir_csv(path)
        except ValueError as exc:
            self._set_status('')
            messagebox.showerror('Load Data', str(exc), parent=self)
            return

        for name, data in spectra.items():
            self._data_pool.append({
                'name':   name,
                'xaxis':  xaxis,
                'data':   data,
                'source': path,
            })
            self._data_lb.insert(tk.END, name)

        self._n_data_label.config(text=f'{len(self._data_pool)} spectra')
        self._set_status(f'Data loaded — {len(spectra)} spectra ({path.name})')

    def _on_build_library(self) -> None:
        """Launch SpeclibViewer (VSWIR mode) to build a custom library; offer to load on close."""
        usgs_path = str(Path(__file__).parent / 'spectral_libraries' / 'usgs_splib07_cvASD.hdf')
        viewer = SpeclibViewer(default_mode='VSWIR', library_path=usgs_path)
        viewer.transient(self)
        viewer.grab_set()
        viewer.title('SpeclibViewer — Build Library (VSWIR)')
        self.wait_window(viewer)

        ans = messagebox.askyesno(
            'Load exported library',
            'Do you want to load an exported library from SpeclibViewer?',
        )
        if ans:
            self._on_load_library()

    def _on_load_library(self) -> None:
        """Open a file chooser and load a spectral library from an HDF5 file."""
        default_dir = str(Path(__file__).parent / 'spectral_libraries')
        path = filedialog.askopenfilename(
            title='Load spectral library',
            initialdir=default_dir,
            filetypes=[('HDF5 files', '*.hdf *.hdf5 *.h5'), ('All files', '*.*')],
        )
        if not path:
            return
        self._load_library_from_path(Path(path))

    def _load_library_from_path(self, path: Path) -> None:
        """
        Load a speclab per-spectrum HDF5 spectral library and replace
        ``_library`` with the new entries.

        The file is read via ``readDVhdf(collapse=False)`` to preserve
        per-spectrum metadata fields.  Filter keyword dropdowns are rebuilt
        dynamically from whatever fields are present in the loaded library.

        Parameters
        ----------
        path : Path
            Path to the HDF5 file.
        """
        self._set_status(f'Reading {path.name}…')
        try:
            raw = readDVhdf(str(path), collapse=False)
        except Exception as exc:
            self._set_status('')
            messagebox.showerror('Load Library', f'Could not read HDF:\n{exc}', parent=self)
            return

        # Each value in raw is a per-spectrum dict with at least 'data' and 'xaxis'.
        # Build self._library keyed by the HDF group id (string).
        library: dict[str, dict] = {}
        for sid, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            if 'data' not in entry or 'xaxis' not in entry:
                continue
            rec = dict(entry)
            rec['source'] = path
            # Normalise display name: prefer 'label', fall back to 'sample_name'/'name'
            if 'label' not in rec:
                rec['label'] = rec.get('sample_name') or rec.get('name') or str(sid)
            library[sid] = rec

        if not library:
            self._set_status('')
            messagebox.showerror('Load Library', 'No valid spectra found in file.', parent=self)
            return

        self._library       = library
        self._lib_plot_items = []
        self._lib_color_idx  = 0
        self._update_filter_fields()   # discover fields and reset filter combos
        self._apply_filters()
        self._sync_lib_plotted_lb()
        self._set_status(f'Library loaded — {len(library)} spectra ({path.name})')

    # -----------------------------------------------------------------------
    # Callbacks — data panel
    # -----------------------------------------------------------------------

    def _on_data_add_single(self) -> None:
        """
        Add each selected data spectrum as an individual plot entry (top of stack).
        """
        indices = list(self._data_lb.curselection())
        if not indices:
            return
        for idx in indices:
            e = self._data_pool[idx]
            item: dict = {
                'kind':   'single',
                'name':   e['name'],
                'xaxis':  e['xaxis'],
                'data':   e['data'],
                'source': e['source'],
                'color':  self._next_data_color(),
            }
            self._smooth_entry(item)
            self._cr_entry(item)
            self._data_plot_items.append(item)
        self._sync_data_plotted_lb()
        self._refresh_plot()

    def _on_data_add_group(self) -> None:
        """
        Bundle selected data spectra into one group entry (top of stack).

        Uses the current group-mode dropdown to set ``display_mode``.
        All selected spectra must share the same wavelength axis length.
        """
        indices = list(self._data_lb.curselection())
        if not indices:
            return
        pool_entries = [self._data_pool[i] for i in indices]

        if len({e['xaxis'].shape[0] for e in pool_entries}) > 1:
            messagebox.showerror(
                'Add as Group',
                'Selected spectra have different wavelength axis lengths '
                'and cannot be grouped without resampling.',
                parent=self,
            )
            return

        self._data_group_count += 1
        mode = ('median'
                if self._data_group_mode_var.get() == 'Median ± std'
                else 'all')

        stacked = np.stack([e['data'] for e in pool_entries])
        item = {
            'kind':         'group',
            'name':         f'Group{self._data_group_count:02d}',
            'xaxis':        pool_entries[0]['xaxis'],
            'data':         stacked,
            'display_mode': mode,
            'source':       [e['source'] for e in pool_entries],
            'color':        self._next_data_color(),
        }
        self._smooth_entry(item)
        self._cr_entry(item)
        self._data_plot_items.append(item)
        self._sync_data_plotted_lb()
        self._refresh_plot()

    def _on_data_select(self, _event=None) -> None:
        pass

    # -----------------------------------------------------------------------
    # Callbacks — library panel
    # -----------------------------------------------------------------------

    def _on_lib_add_single(self) -> None:
        """
        Add each selected library spectrum as an individual plot entry (bottom of stack).

        Entries are prepended so that newly added items appear below existing ones.
        """
        indices = list(self._lib_lb.curselection())
        if not indices:
            return
        # Prepend in reverse selection order so visual order matches selection order.
        for idx in reversed(indices):
            e = self._library[self._filtered_ids[idx]]
            item = {
                'kind':   'single',
                'name':   e.get('label') or e.get('sample_name') or e.get('name') or self._filtered_ids[idx],
                'xaxis':  e['xaxis'],
                'data':   e['data'],
                'source': e['source'],
                'color':  self._next_lib_color(),
            }
            self._smooth_entry(item)
            self._cr_entry(item)
            self._lib_plot_items.insert(0, item)
        self._sync_lib_plotted_lb()
        self._refresh_plot()

    def _on_lib_add_group(self) -> None:
        """
        Bundle selected library spectra into one group entry (bottom of stack).

        Uses the current group-mode dropdown to set ``display_mode``.
        All selected spectra must share the same wavelength axis length.
        """
        indices = list(self._lib_lb.curselection())
        if not indices:
            return
        lib_entries = [self._library[self._filtered_ids[i]] for i in indices]

        if len({e['xaxis'].shape[0] for e in lib_entries}) > 1:
            messagebox.showerror(
                'Add as Group',
                'Selected spectra have different wavelength axis lengths '
                'and cannot be grouped without resampling.',
                parent=self,
            )
            return

        self._lib_group_count += 1
        mode = ('median'
                if self._lib_group_mode_var.get() == 'Median ± std'
                else 'all')

        stacked = np.stack([e['data'] for e in lib_entries])
        item = {
            'kind':         'group',
            'name':         f'Group{self._lib_group_count:02d}',
            'xaxis':        lib_entries[0]['xaxis'],
            'data':         stacked,
            'display_mode': mode,
            'source':       [e['source'] for e in lib_entries],
            'color':        self._next_lib_color(),
        }
        self._smooth_entry(item)
        self._cr_entry(item)
        self._lib_plot_items.insert(0, item)
        self._sync_lib_plotted_lb()
        self._refresh_plot()

    def _on_lib_select(self, _event=None) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch() -> None:
    app = ReflectanceVSWIR()
    app.mainloop()


if __name__ == '__main__':
    launch()
