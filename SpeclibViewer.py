#!/usr/bin/env python3
"""
SpeclibViewer — GUI browser and album builder for TIR and VNIR spectral libraries.

Supports two spectral modes, detected automatically from the loaded library:
  TIR  — thermal infrared (wavenumber cm⁻¹), emissivity axis, TES/ASTER/THEMIS
          resampling, atmospheric opacity overlay.
  VNIR — visible/near-infrared (wavelength nm), reflectance axis, no instrument
          resampling (heterogeneous xaxes expected).

Layout
------
Top bar  : Library selector dropdown | Load Library || Reset Album | Export Album
Left     : dynamic filter rows (key + value comboboxes, discovered from library)
           → available-spectra listbox → action button
           → current-album listbox → album action buttons
Right    : ttk.Notebook
             Active tab: single-spectrum plot for the currently active spectrum
                            + metadata treeview below
             Album tab    : mode toggle (Stacked / Individual)
                            + sampling dropdown (TIR only)
                            + atmosphere overlay (TIR only)
                            + matplotlib figure + navigation toolbar
                            + navigation bar (◀  Spectrum x/n  ▶)
             Summary tab  : pandas-backed metadata table for all album spectra

Data model
----------
_full_library   : {spec_id: entry}  — full HDF, loaded once at launch
_current_album  : {spec_id: entry}  — user selection, empty at launch
_extra_libs     : {display_name: path}  — additional HDF files added at runtime
_browse_source  : reference to whichever source is selected in the top dropdown
_active_sid     : spec_id of the currently displayed spectrum (None if none)
_mode           : 'LWIR' | 'VSWIR' — set automatically on library load
"""

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

# Allow running directly as a script (`python speclab/SpeclibViewer.py`) in
# addition to the normal entry points (`speclib-viewer`, `python -m speclab.SpeclibViewer`).
if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'speclab'

from .functions import resample_spectrum, load_instrument_grids, insert_plot_gaps
from .utils import readDVhdf, saveDVhdf
from .config import get_config
from . import __version__

import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size':               12,
    'lines.linewidth':         1.0,
    'axes.prop_cycle':         plt.cycler(color=plt.cm.Dark2.colors),
    'xtick.direction':         'in',
    'xtick.top':               True,
    'ytick.direction':         'in',
    'xtick.labelsize':         11,
    'ytick.right':             True,
    'ytick.labelsize':         11,
    'axes.grid':               True,
    'axes.axisbelow':          True,
    'axes.labelsize':          12,
    'axes.titlesize':          12,
    'grid.linestyle':          '--',
    'axes.formatter.limits':   (-4, 4),
    'errorbar.capsize':        2,
})

def _scalar(v: object) -> object:
    """
    Convert any numpy scalar or array to a Python native type.

    Arrays are squeezed; if the result is still multi-dimensional the first
    element is used as a fallback.  This guarantees that metadata fields
    (everything except ``data`` and ``xaxis``) are always Python-native and
    safe to use in format strings and Tkinter widgets.
    """
    if isinstance(v, np.ndarray):
        s = v.squeeze()
        raw = s.item() if s.ndim == 0 else s.flat[0]
        if hasattr(raw, 'item'):
            raw = raw.item()
        if isinstance(raw, (bytes, np.bytes_)):
            return raw.decode('utf-8')
        return raw
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8')
    return v


def _load_hdf(path: str) -> dict:
    """
    Load a spectral library HDF5 file into the ``{spec_id: entry}`` album
    format used internally by the viewer.

    Delegates I/O to :func:`utils.readDVhdf` and handles two on-disk layouts:

    * **Per-spectrum** (``saveDVhdf`` / SpeclibViewer export) — one sub-group
      per spectrum at the top level.  String integer keys are cast to ``int``
      spec-IDs.
    * **Grouped** (``makeASUspeclib_dev`` / DaVinci native) — a shared
      ``xaxis`` and a ``data`` matrix ``(n_spec, n_pts)`` at the top level.
      Per-spectrum metadata arrays are distributed across individual entries.

    Scalar metadata fields (0-d numpy arrays, numpy integers/floats, bytes)
    are converted to Python native types so they are safe to use in format
    strings and Tkinter widgets.

    Parameters
    ----------
    path : str
        Path to the HDF5 spectral library file.

    Returns
    -------
    dict
        Album dict ``{spec_id: {'data': np.ndarray, 'xaxis': np.ndarray, ...}}``.

    Raises
    ------
    ValueError
        If the loaded dict does not match either recognised layout.
    """
    raw = readDVhdf(path, collapse=False)

    def _expand_group(grp: dict, start_id: int, album: dict) -> int:
        """Expand a grouped sub-dict (shared xaxis + 2-D data) into album."""
        xaxis  = np.asarray(grp['xaxis'], dtype=np.float64)
        data   = np.atleast_2d(np.asarray(grp['data'], dtype=np.float64))
        n_spec = data.shape[0]
        for i in range(n_spec):
            entry: dict = {'data': data[i], 'xaxis': xaxis}
            for key, val in grp.items():
                if key in ('xaxis', 'data'):
                    continue
                if isinstance(val, np.ndarray) and len(val) == n_spec:
                    entry[key] = _scalar(val[i])
                else:
                    entry[key] = _scalar(val)
            album[start_id + i] = entry
        return start_id + n_spec

    # All top-level values are sub-dicts
    if all(isinstance(v, dict) for v in raw.values()):
        first = next(iter(raw.values()))

        # Nested-grouped format: each sub-dict holds xaxis + 2-D data matrix
        if 'xaxis' in first and 'data' in first and np.ndim(first['data']) == 2:
            album: dict = {}
            next_id = 0
            for grp in raw.values():
                next_id = _expand_group(grp, next_id, album)
            return album

        # Per-spectrum format: each sub-dict is a single spectrum
        album = {}
        for k, entry in raw.items():
            try:
                spec_id: int | str = int(k)
            except (ValueError, TypeError):
                spec_id = k
            album[spec_id] = {
                field: (val if field in ('data', 'xaxis') else _scalar(val))
                for field, val in entry.items()
            }
        return album

    # Flat-grouped format: shared xaxis + data matrix at the top level
    if 'xaxis' in raw and 'data' in raw:
        album = {}
        _expand_group(raw, 0, album)
        return album

    raise ValueError(
        f"Cannot load '{path}' as a spectral library: unrecognised HDF5 layout."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL          = 'All'
_SRC_FULL    = 'Full Library'
_SRC_ALBUM   = 'Current Album'
MAX_STACKED  = 10   # spectra per page in stacked mode
_N_FILTER_ROWS = 4  # number of dynamic filter rows

# VNIR export resolution options
_ASD_1NM_XAXIS: np.ndarray = np.arange(350.0, 2501.0, 1.0)   # 2151 pts, nm
_VNIR_SAMPLING_OPTS: dict[str, dict] = {
    'Original': {'xaxis': None,           'label': 'Original resolution'},
    'ASD 1nm':  {'xaxis': _ASD_1NM_XAXIS, 'label': 'ASD 1 nm  (350–2500 nm)'},
}

# Fields excluded from dynamic filter discovery
_FILTER_NON_FIELDS: frozenset[str] = frozenset({
    'data', 'xaxis', 'name', 'label', 'sample_name',
    'source', 'hdf_group', 'spec_id', 'spectral_mode',
    'x_unit', 'y_unit',
})

# Preferred filter pre-selection order per mode (display names)
_TIR_DEFAULT_FILTERS:  list[str] = ['Type', 'Subgroup', 'Category', 'Quality']
_VNIR_DEFAULT_FILTERS: list[str] = []   # use discovery order (fewest-unique first)

# Display-name overrides for discovered filter fields (both TIR and VNIR)
_FILTER_FIELD_DISPLAY: dict[str, str] = {
    # TIR / ASU library fields
    'group':              'Type',
    'type_subgroup':      'Subgroup',
    'quality':            'Quality',
    'category':           'Category',
    'particle_size':      'Particle Size',
    'instrument':         'Instrument',
    'source_lab':         'Source Lab',
    # VNIR / USGS library fields
    'chapter':            'Chapter',
    'mineral_name':       'Mineral',
    'mineral_type':       'Mineral Type',
    'mineral':            'Mineral (full)',
    'meas_type':          'Meas. Type',
    'spectrometer':       'Spectrometer',
    'original_donor':     'Donor',
    'source_library':     'Library',
    'collection_locality':'Locality',
    # VNIR / CRISM library fields
    'type':               'Instr. Type',
    'body':               'Body',
    'material':           'Material',
    'mineral_family':     'Min. Family',
    'reference':          'Reference',
    'current_location':   'Repository',
    'collection_location':'Origin',
}

# Source library strings that identify VNIR libraries
_VNIR_SOURCE_LIBS: frozenset[str] = frozenset({
    'USGS splib07b_cvASD',
    'CRISM Spectral Library',
})

# Ordered metadata groups for the Info tab — TIR libraries (ASU).
# Each entry: (group_display_name, [(hdf_field, display_label), ...])
# Fields with nan/empty values are suppressed automatically.
_INFO_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ('Identification', [
        ('spec_id',               'Spectrum ID'),
        ('sample_id',             'Sample ID'),
        ('sample_name',           'Sample Name'),
        ('chemical_formula',      'Chemical Formula'),
        ('dana_mineral_number',   'Dana Number'),
        ('dana_class_description','Dana Class'),
    ]),
    ('Classification', [
        ('group',        'Mineral Group'),
        ('category',     'Category'),
        ('type_subgroup','Type / Subgroup'),
        ('particle_size','Particle Size'),
    ]),
    ('Provenance', [
        ('source',              'Source'),
        ('source_lab',          'Source Lab'),
        ('owner',               'Owner'),
        ('citation',            'Citation'),
        ('collection_locality', 'Collection Locality'),
        ('sample_location',     'Sample Location'),
        ('latitude',            'Latitude'),
        ('longitude',           'Longitude'),
    ]),
    ('Measurement', [
        ('instrument',             'Instrument'),
        ('resolution',             'Resolution (cm\u207b\xb9)'),
        ('quality',                'Quality'),
        ('analysis_date',          'Analysis Date'),
        ('spectral_analysis_person','Analyst'),
        ('wavenum_range_low',      'Wavenumber Min (cm\u207b\xb9)'),
        ('wavenum_range_high',     'Wavenumber Max (cm\u207b\xb9)'),
    ]),
    # Collapsed by default — instrument housekeeping / calibration values
    ('Measurement Details', [
        ('chamber_temperature', 'Chamber Temp (\u00b0C)'),
        ('hotbb_temperature',   'Hot BB Temp (\u00b0C)'),
        ('warmbb_temperature',  'Warm BB Temp (\u00b0C)'),
        ('sample_temperature',  'Sample Temp (\u00b0C)'),
        ('field_of_view',       'Field of View'),
        ('hbb',                 'HBB'),
        ('wbb',                 'WBB'),
        ('radiance',            'Radiance'),
        ('raw',                 'Raw'),
        ('response',            'Response'),
    ]),
]

_INFO_COLLAPSED_BY_DEFAULT = {'Measurement Details'}

# Metadata groups for VNIR libraries (USGS, CRISM, etc.)
_VNIR_INFO_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ('Identification', [
        ('label',            'Label'),
        ('sample_name',      'Sample Name'),
        ('specimen_name',    'Specimen Name'),
        ('specimen_id',      'Specimen ID'),
        ('crism_id',         'CRISM ID'),
        ('mineral_name',     'Mineral'),
        ('sample_id',        'Sample ID'),
        ('chemical_formula', 'Formula'),
    ]),
    ('Classification', [
        ('chapter',        'Chapter (USGS)'),
        ('mineral_type',   'Mineral Type'),
        ('mineral',        'Mineral (full)'),
        ('body',           'Body'),
        ('material',       'Material'),
        ('mineral_family', 'Min. Family'),
        ('type',           'Instr. Type'),
        ('spectrometer',   'Spectrometer'),
        ('meas_type',      'Meas. Type'),
    ]),
    ('Provenance', [
        ('source_library',       'Library'),
        ('collection_locality',  'Collection Locality'),
        ('collection_location',  'Collection Location'),
        ('original_donor',       'Donor'),
        ('current_location',     'Repository'),
        ('reference',            'Reference'),
    ]),
]

_VNIR_SUMMARY_FIELDS: list[tuple[str, str]] = [
    ('label',          'Label'),
    ('mineral_name',   'Mineral'),
    ('mineral_family', 'Family'),
    ('body',           'Body'),
    ('material',       'Material'),
    ('type',           'Instr. Type'),
    ('chapter',        'Chapter'),
    ('source_library', 'Library'),
]

# Instrument grids whose xaxis is in µm (wavelength) rather than cm⁻¹
_WL_GRIDS = {'THEMIS', 'ASTER', 'MASTER', 'TIMS'}

_COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


# Atmospheric opacity/window spans for overlay.
# Each entry: list of (wn_lo, wn_hi, colour, label) in cm⁻¹.
# For wavelength-based plots, wn values are converted to µm on the fly.
_ATMOSPHERE_SPANS = {
    'None': [],
    'Mars': [
        (507.88, 825.31, '#ffcccc', 'Atmospheric opacity (CO\u2082)'),
    ],
    # Three opacity bands for Earth; the window sits at 715–1250 cm⁻¹.
    # Sentinel values (10 and 50 000 cm⁻¹) extend spans to the plot edges.
    'Earth': [
        (10.0,   715.0,  '#cce5ff', 'Atmospheric opacity'),
        (1250.0, 2000.0, '#cce5ff', 'Atmospheric opacity'),
        (3333.0, 50000., '#cce5ff', 'Atmospheric opacity'),
    ],
}

# Metadata columns shown in the Summary tab table.
# Each entry: (hdf_field, display_label)
_SUMMARY_FIELDS: list[tuple[str, str]] = [
    ('spec_id',          'ID'),
    ('sample_name',      'Sample Name'),
    ('chemical_formula', 'Formula'),
    ('group',            'Group'),
    ('category',         'Category'),
    ('type_subgroup',    'Subgroup'),
    ('particle_size',    'Particle Size'),
    ('quality',          'Quality'),
    ('source',           'Source'),
    ('instrument',       'Instrument'),
    ('resolution',       'Resolution'),
]


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def _detect_library_mode(library: dict) -> str:
    """
    Detect whether a loaded spectral library uses TIR or VNIR axes.

    Priority
    --------
    1. ``x_unit`` field: ``'nm'`` → VNIR; ``'um'`` or ``'cm-1'`` → TIR.
    2. ``y_unit`` field: ``'reflectance'`` → VNIR; ``'emissivity'`` / ``'transmittance'`` → TIR.
    3. ``spectral_mode`` field (legacy tag): ``'VSWIR'`` or ``'LWIR'``.
    4. ``source_library`` membership in :data:`_VNIR_SOURCE_LIBS` (legacy).

    Returns
    -------
    'LWIR' or 'VSWIR'
    """
    for entry in library.values():
        x_unit = str(entry.get('x_unit', '')).strip().lower()
        if x_unit == 'nm':
            return 'VSWIR'
        if x_unit in ('um', 'cm-1'):
            return 'LWIR'

        y_unit = str(entry.get('y_unit', '')).strip().lower()
        if y_unit == 'reflectance':
            return 'VSWIR'
        if y_unit in ('emissivity', 'transmittance'):
            return 'LWIR'

        mode = str(entry.get('spectral_mode', '')).strip().upper()
        if mode in ('LWIR', 'VSWIR'):
            return mode

        src = str(entry.get('source_library', '')).strip()
        if src in _VNIR_SOURCE_LIBS:
            return 'VSWIR'

    return 'LWIR'


# ---------------------------------------------------------------------------
# Sampling option registry
# ---------------------------------------------------------------------------

def _load_sampling_options() -> dict:
    """
    Build the display-name registry of sampling grids for the GUI dropdown.

    Returns
    -------
    dict
        {display_name: {'xaxis': ndarray | None, 'is_wl': bool, 'xlabel': str}}
        xaxis=None means 'Original' (no resampling).
    """
    opts = {
        'Original': {
            'xaxis': None, 'is_wl': False,
            'xlabel': 'Wavenumber (cm\u207b\xb9)',
        },
        'Speclab1 (200\u20132001 cm\u207b\xb9, 2 cm\u207b\xb9) [default]': {
            'xaxis': np.arange(200, 2002, 2), 'is_wl': False,
            'xlabel': 'Wavenumber (cm\u207b\xb9)',
        },
        'Speclab2 (200\u20134001 cm\u207b\xb9, 2 cm\u207b\xb9)': {
            'xaxis': np.arange(200, 4002, 2), 'is_wl': False,
            'xlabel': 'Wavenumber (cm\u207b\xb9)',
        },
    }

    # Display name → instrument key in load_instrument_grids()
    _display_names = {
        'tessingle': 'TES Single',
        'tesdouble': 'TES Double',
        'tes73':     'TES 73',
        'minites':   'MiniTES',
        'microlab':  'MicroLab',
        'themis':    'THEMIS',
        'aster':     'ASTER',
        'master':    'MASTER',
        'tims':      'TIMS',
    }

    grids = load_instrument_grids()
    for key, display in _display_names.items():
        if key in grids:
            opts[display] = grids[key]

    return opts


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class SpeclibViewer(tk.Toplevel):

    def __init__(self, master: tk.Misc | None = None, *,
                 default_mode: str = 'LWIR',
                 library_path: str | None = None) -> None:
        # Standalone mode: no existing Tk root — create a hidden one so Toplevel has a parent.
        # Embedded mode: a Tk root already exists; pass master=None so Toplevel.__init__
        # picks up _default_root automatically.
        if master is None and tk._default_root is None:
            self._standalone_root = tk.Tk()
            self._standalone_root.withdraw()
            master = self._standalone_root
        else:
            self._standalone_root = None
        super().__init__(master)
        self.title(f'SpeclibViewer  v{__version__}')
        self.geometry('1400x860')
        self.minsize(1000, 620)

        # Data state
        self._full_library:   dict = {}
        self._current_album:  dict = {}
        self._extra_libs:     dict = {}
        self._browse_source:  dict = {}
        self._browse_name:    str  = _SRC_FULL
        self._filtered_ids:   list = []
        self._active_sid           = None
        self._mode:           str  = default_mode   # 'LWIR' | 'VSWIR'
        self._init_library:   str | None = library_path

        # Plot state
        self._plot_mode      = 'stacked'   # 'stacked' | 'individual'
        self._indiv_idx      = 0
        self._stacked_page   = 0
        self._sampling_opts  = _load_sampling_options()
        self._current_samp   = 'Original'
        self._atm_var        = tk.StringVar(value='None')
        self._secax          = None        # secondary (top) x-axis — album figure
        self._active_secax   = None        # secondary (top) x-axis — active figure

        # Filter state (dynamic — populated after library load)
        self._lib_filter_fields: dict[str, str] = {}  # display_name → field_name
        self._filter_key_vars:   list[tk.StringVar] = []
        self._filter_val_vars:   list[tk.StringVar] = []
        self._filter_key_combos: list[ttk.Combobox] = []
        self._filter_val_combos: list[ttk.Combobox] = []

        self._build_ui()

        # Keyboard navigation for Individual mode
        self.bind('<Left>',  lambda _e: self._on_prev())
        self.bind('<Right>', lambda _e: self._on_next())

        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._discover_libraries()

    def _on_close(self) -> None:
        plt.close('all')
        self.destroy()
        if self._standalone_root is not None:
            self._standalone_root.destroy()

    # -----------------------------------------------------------------------
    # Library discovery
    # -----------------------------------------------------------------------

    # Package-bundled fallback libraries (relative to this file's directory)
    _PKG_LIB_DIR = Path(__file__).parent / 'spectral_libraries'
    _MODE_DEFAULT_LIB = {
        'LWIR':  'ASU_speclib_full_SV.hdf',
        'VSWIR': 'usgs_splib07_cvASD.hdf',
    }

    def _discover_libraries(self) -> None:
        """Load default library and populate extra-libs from spectral_libraries_dir."""
        cfg = get_config()

        # Priority 1: explicit library_path passed at construction
        default = self._init_library
        # Priority 2: user config default_library
        if not default:
            default = cfg.get('default_library') or ''
        # Priority 3: mode-specific package-bundled fallback
        if not default or not Path(default).is_file():
            fallback_name = self._MODE_DEFAULT_LIB.get(self._mode, '')
            if fallback_name:
                candidate = self._PKG_LIB_DIR / fallback_name
                if candidate.is_file():
                    default = str(candidate)

        if default and Path(default).is_file():
            self._load_full_library(default)

        # Extra libs: scan spectral_libraries_dir for additional HDF files.
        lib_dir = cfg.get('spectral_libraries_dir') or str(self._PKG_LIB_DIR)
        if not lib_dir or not Path(lib_dir).is_dir():
            return

        patterns = ('*.hdf', '*.h5', '*.hdf5')
        found = sorted(
            p for pat in patterns for p in Path(lib_dir).glob(pat)
        )
        for p in found:
            stem = p.stem
            name = stem
            idx  = 2
            while name in self._extra_libs or name in (_SRC_FULL, _SRC_ALBUM):
                name = f'{stem} ({idx})'
                idx += 1
            self._extra_libs[name] = str(p)

        if found:
            self._update_src_combo()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Top toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))

        ttk.Label(toolbar, text='Library:').pack(side=tk.LEFT, padx=(0, 4))
        self._src_var   = tk.StringVar(value=_SRC_FULL)
        self._src_combo = ttk.Combobox(
            toolbar, textvariable=self._src_var,
            state='readonly', width=28, values=[_SRC_FULL, _SRC_ALBUM],
        )
        self._src_combo.pack(side=tk.LEFT, padx=(0, 2))
        self._src_combo.bind('<<ComboboxSelected>>',
                             lambda _e: self._on_source_changed())

        ttk.Button(toolbar, text='Load Library',
                   command=self._on_load).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Button(toolbar, text='Reset Album',
                   command=self._on_reset_album).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text='Export Album',
                   command=self._on_export).pack(side=tk.LEFT, padx=2)

        self._status_label = ttk.Label(toolbar, text='', foreground='gray')
        self._status_label.pack(side=tk.LEFT, padx=14)

        # Main paned window
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        left = ttk.Frame(paned, width=400)
        paned.add(left, weight=0)

        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_left_panel(left)
        self._build_notebook(right)

    # -----------------------------------------------------------------------
    # Left panel
    # -----------------------------------------------------------------------

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        # Dynamic filter rows
        filter_frame = ttk.LabelFrame(parent, text='Filters', padding=4)
        filter_frame.pack(fill=tk.X, padx=4, pady=(4, 2))

        for i in range(_N_FILTER_ROWS):
            row_frame = ttk.Frame(filter_frame)
            row_frame.pack(fill=tk.X, pady=1)

            key_var = tk.StringVar(value='')
            val_var = tk.StringVar(value=ALL)

            key_cb = ttk.Combobox(
                row_frame, textvariable=key_var,
                values=[], state='disabled', width=12,
            )
            key_cb.pack(side=tk.LEFT, padx=(0, 4))

            val_cb = ttk.Combobox(
                row_frame, textvariable=val_var,
                values=[ALL], state='disabled', width=18,
            )
            val_cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

            key_cb.bind('<<ComboboxSelected>>',
                        lambda _e, idx=i: self._on_filter_key_changed(idx))
            val_cb.bind('<<ComboboxSelected>>', lambda _e: self._apply_filters())

            self._filter_key_vars.append(key_var)
            self._filter_val_vars.append(val_var)
            self._filter_key_combos.append(key_cb)
            self._filter_val_combos.append(val_cb)

        # Available spectra listbox
        avail_frame = ttk.LabelFrame(parent, text='Available Spectra', padding=4)
        avail_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        self._n_avail_label = ttk.Label(avail_frame, text='0 spectra')
        self._n_avail_label.pack(anchor=tk.W)

        lb_frame = ttk.Frame(avail_frame)
        lb_frame.pack(fill=tk.BOTH, expand=True)

        vs = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        hs = ttk.Scrollbar(lb_frame, orient=tk.HORIZONTAL)
        self._avail_lb = tk.Listbox(
            lb_frame, selectmode=tk.EXTENDED,
            yscrollcommand=vs.set, xscrollcommand=hs.set,
            activestyle='dotbox', height=16, font=('Tk', 12),
        )
        vs.config(command=self._avail_lb.yview)
        hs.config(command=self._avail_lb.xview)
        self._avail_lb.grid(row=0, column=0, sticky=tk.NSEW)
        vs.grid(row=0, column=1, sticky=tk.NS)
        hs.grid(row=1, column=0, sticky=tk.EW)
        lb_frame.rowconfigure(0, weight=1)
        lb_frame.columnconfigure(0, weight=1)
        self._avail_lb.bind('<<ListboxSelect>>', self._on_avail_selection_changed)

        btn_frame = ttk.Frame(avail_frame)
        btn_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(btn_frame, text='Add to Album',
                   command=self._on_add_to_album).pack(side=tk.LEFT, padx=2)

        # Current album listbox
        album_frame = ttk.LabelFrame(parent, text='Current Album', padding=4)
        album_frame.pack(fill=tk.BOTH, padx=4, pady=(2, 4))

        self._n_album_label = ttk.Label(album_frame, text='0 spectra')
        self._n_album_label.pack(anchor=tk.W)

        af = ttk.Frame(album_frame)
        af.pack(fill=tk.BOTH, expand=True)

        avs = ttk.Scrollbar(af, orient=tk.VERTICAL)
        ahs = ttk.Scrollbar(af, orient=tk.HORIZONTAL)
        self._album_lb = tk.Listbox(
            af, selectmode=tk.EXTENDED,
            yscrollcommand=avs.set, xscrollcommand=ahs.set,
            activestyle='dotbox', height=7, font=('TkFixedFont', 13),
        )
        avs.config(command=self._album_lb.yview)
        ahs.config(command=self._album_lb.xview)
        self._album_lb.grid(row=0, column=0, sticky=tk.NSEW)
        avs.grid(row=0, column=1, sticky=tk.NS)
        ahs.grid(row=1, column=0, sticky=tk.EW)
        af.rowconfigure(0, weight=1)
        af.columnconfigure(0, weight=1)

        alb_btn = ttk.Frame(album_frame)
        alb_btn.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(alb_btn, text='Remove Selected',
                   command=self._on_album_remove).pack(side=tk.LEFT, padx=2)

    # -----------------------------------------------------------------------
    # Right panel — notebook
    # -----------------------------------------------------------------------

    def _build_notebook(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True)

        active_tab  = ttk.Frame(nb)
        album_tab   = ttk.Frame(nb)
        summary_tab = ttk.Frame(nb)

        nb.add(active_tab,  text='  Active  ')
        nb.add(album_tab,   text='  Album  ')
        nb.add(summary_tab, text='  Summary  ')

        self._build_active_tab(active_tab)
        self._build_album_tab(album_tab)
        self._build_summary_tab(summary_tab)

    def _build_active_tab(self, parent: ttk.Frame) -> None:
        """
        Active tab: single-spectrum or multi-spectrum stacked figure on top,
        metadata treeview below.  A vertical PanedWindow lets the user resize the split.
        """
        # --- Mode toggle row ---
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=6, pady=(4, 0))
        self._active_mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(ctrl, text='Single',  variable=self._active_mode_var,
                        value='single',  command=self._on_active_mode_change,
                        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(ctrl, text='Stacked', variable=self._active_mode_var,
                        value='stacked', command=self._on_active_mode_change,
                        ).pack(side=tk.LEFT)

        paned = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # --- Top: single-spectrum plot ---
        plot_frame = ttk.Frame(paned)
        paned.add(plot_frame, weight=1)

        self._active_fig = Figure(dpi=100)
        self._active_ax  = self._active_fig.add_subplot(111)
        self._active_ax.set_xlabel('Wavenumber (cm⁻¹)')
        self._active_ax.set_ylabel('Emissivity')
        self._active_ax.set_xlim(4000, 200)
        self._active_ax.set_ylim(0, 1.05)
        self._active_fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)
        self._active_secax = self._add_top_axis(self._active_ax)

        self._active_canvas = FigureCanvasTkAgg(self._active_fig, master=plot_frame)
        self._active_canvas.draw()
        self._active_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        active_nav = NavigationToolbar2Tk(self._active_canvas, plot_frame)
        active_nav.update()

        # --- Bottom: metadata treeview ---
        info_frame = ttk.Frame(paned)
        paned.add(info_frame, weight=1)

        vs = ttk.Scrollbar(info_frame, orient=tk.VERTICAL)
        # show='tree headings': exposes the tree column (#0) for expand/collapse
        self._info_tree = ttk.Treeview(
            info_frame, columns=('field', 'value'),
            show='tree headings',
            yscrollcommand=vs.set, selectmode='none',
        )
        vs.config(command=self._info_tree.yview)

        self._info_tree.heading('#0',    text='')
        self._info_tree.heading('field', text='Field')
        self._info_tree.heading('value', text='Value')
        self._info_tree.column('#0',    width=18, minwidth=18, stretch=False)
        self._info_tree.column('field', width=210, stretch=False)
        self._info_tree.column('value', width=480, stretch=True)

        self._info_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vs.pack(side=tk.RIGHT, fill=tk.Y)

        self._info_tree.tag_configure(
            'header', background='#dce3f0', font=('TkDefaultFont', 9, 'bold'))
        self._info_tree.tag_configure('odd',  background='#f5f5f5')
        self._info_tree.tag_configure('even', background='#ffffff')

    def _build_album_tab(self, parent: ttk.Frame) -> None:
        # --- Controls row ---
        ctrl = ttk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=6, pady=(4, 2))

        # Mode toggle
        ttk.Label(ctrl, text='Mode:').pack(side=tk.LEFT, padx=(0, 4))
        self._mode_var = tk.StringVar(value='stacked')
        ttk.Radiobutton(ctrl, text='Stacked',    variable=self._mode_var,
                        value='stacked',    command=self._on_mode_change
                        ).pack(side=tk.LEFT)
        ttk.Radiobutton(ctrl, text='Individual', variable=self._mode_var,
                        value='individual', command=self._on_mode_change
                        ).pack(side=tk.LEFT, padx=(0, 16))

        # Sampling section (TIR only — hidden for VNIR)
        self._samp_sep = ttk.Separator(ctrl, orient=tk.VERTICAL)
        self._samp_sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._samp_label = ttk.Label(ctrl, text='Sampling:')
        self._samp_label.pack(side=tk.LEFT, padx=(0, 4))
        self._samp_var = tk.StringVar(value='Original')
        self._samp_combo = ttk.Combobox(
            ctrl, textvariable=self._samp_var,
            state='readonly', width=30,
            values=list(self._sampling_opts.keys()),
        )
        self._samp_combo.pack(side=tk.LEFT)
        self._samp_combo.bind('<<ComboboxSelected>>', lambda _e: self._on_sampling_change())

        # Atmosphere section (TIR only — hidden for VNIR)
        self._atm_sep = ttk.Separator(ctrl, orient=tk.VERTICAL)
        self._atm_sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._atm_label = ttk.Label(ctrl, text='Atm. Opacity:')
        self._atm_label.pack(side=tk.LEFT, padx=(0, 4))
        self._atm_combo = ttk.Combobox(
            ctrl, textvariable=self._atm_var,
            state='readonly', width=5,
            values=list(_ATMOSPHERE_SPANS.keys()),
        )
        self._atm_combo.pack(side=tk.LEFT)
        self._atm_combo.bind('<<ComboboxSelected>>', lambda _e: self._draw_album())

        # --- Navigation bar (both modes) ---
        self._nav_frame = ttk.Frame(parent)
        self._nav_frame.pack(pady=(0, 4))  # visible by default (stacked is default mode)

        self._nav_prev  = ttk.Button(self._nav_frame, text='◀',
                                     width=3, command=self._on_prev)
        self._nav_prev.pack(side=tk.LEFT, padx=4)
        self._nav_label = ttk.Label(self._nav_frame, text='— / —',
                                    anchor=tk.CENTER, width=16)
        self._nav_label.pack(side=tk.LEFT, padx=4)
        self._nav_next  = ttk.Button(self._nav_frame, text='▶',
                                     width=3, command=self._on_next)
        self._nav_next.pack(side=tk.LEFT, padx=4)

        # --- Matplotlib figure ---
        self._fig = Figure(dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_xlabel('Wavenumber (cm⁻¹)')
        self._ax.set_ylabel('Emissivity')
        self._ax.set_xlim(4000, 200)
        self._ax.set_ylim(0, 1.05)
        self._fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)
        self._secax = self._add_top_axis(self._ax)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = NavigationToolbar2Tk(self._canvas, parent)
        nav.update()

    def _build_summary_tab(self, parent: ttk.Frame) -> None:
        """
        Summary tab: flat Treeview table backed by a pandas DataFrame.
        One row per album spectrum; columns are the fields in _SUMMARY_FIELDS.
        """
        self._summary_parent = parent
        self._summary_frame  = None
        self._summary_tree   = None
        self._rebuild_summary_tree(_SUMMARY_FIELDS)

    # -----------------------------------------------------------------------
    # Library loading / source switching
    # -----------------------------------------------------------------------

    def _load_full_library(self, path: str) -> None:
        try:
            self._full_library  = _load_hdf(path)
            self._browse_source = self._full_library
            self._browse_name   = _SRC_FULL
            self._mode = _detect_library_mode(self._full_library)
            self._update_status()
            self._apply_mode_ui()
            self._update_filter_fields()
            self._apply_filters()
        except Exception as exc:
            messagebox.showerror('Load error', str(exc))

    def _set_browse_source(self, name: str) -> None:
        self._browse_name = name
        if name == _SRC_FULL:
            self._browse_source = self._full_library
        elif name == _SRC_ALBUM:
            self._browse_source = self._current_album
        else:
            path = self._extra_libs[name]
            try:
                self._browse_source = _load_hdf(path)
            except Exception as exc:
                messagebox.showerror('Load error', str(exc))
                self._src_var.set(_SRC_FULL)
                self._browse_source = self._full_library
                self._browse_name   = _SRC_FULL

        self._mode = _detect_library_mode(self._browse_source)
        self._update_status()
        self._apply_mode_ui()
        self._update_filter_fields()
        self._apply_filters()

    def _update_status(self) -> None:
        n = len(self._browse_source)
        self._status_label.config(
            text=f'{self._browse_name}  ({n} spectra)',
            foreground='black' if n else 'gray',
        )

    def _update_src_combo(self) -> None:
        self._src_combo['values'] = (
            [_SRC_FULL, _SRC_ALBUM] + list(self._extra_libs.keys())
        )

    # -----------------------------------------------------------------------
    # Mode UI — show/hide TIR-only controls
    # -----------------------------------------------------------------------

    def _apply_mode_ui(self) -> None:
        """Show or hide TIR-only controls and reset plot axes for self._mode."""
        tir_widgets = [
            self._samp_sep, self._samp_label, self._samp_combo,
            self._atm_sep,  self._atm_label,  self._atm_combo,
        ]
        if self._mode == 'LWIR':
            for w in tir_widgets:
                w.pack_configure()           # restore
        else:
            for w in tir_widgets:
                w.pack_forget()

        # Reset sampling to Original for VNIR (instrument grids are TIR-only)
        if self._mode == 'VSWIR':
            self._samp_var.set('Original')
            self._current_samp = 'Original'

        # Update both matplotlib axes to reflect the new mode
        xlabel = self._current_xlabel()
        ylabel = self._ylabel()
        lo, hi = self._default_xlim()
        for ax, secax_attr, canvas in (
            (self._active_ax, '_active_secax', self._active_canvas),
            (self._ax,        '_secax',        self._canvas),
        ):
            ax.cla()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0, 1.05)
            old_secax = getattr(self, secax_attr, None)
            if old_secax is not None:
                try:
                    old_secax.remove()
                except Exception:
                    pass
            setattr(self, secax_attr, self._add_top_axis(ax))
            canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Dynamic filter system
    # -----------------------------------------------------------------------

    def _update_filter_fields(self) -> None:
        """
        Scan the current browse source and populate the dynamic filter rows.

        Discovers all string metadata fields with 2–200 unique non-empty values,
        sorted fewest-unique first.  The first _N_FILTER_ROWS are pre-selected.
        """
        field_values: dict[str, set] = {}
        _suppress = {'nan', '-', '0', '', 'none', 'n/a'}
        for entry in self._browse_source.values():
            for k, v in entry.items():
                if k in _FILTER_NON_FIELDS:
                    continue
                sv = str(v).strip()
                if sv.lower() not in _suppress:
                    field_values.setdefault(k, set()).add(sv)

        valid = {k: v for k, v in field_values.items() if 2 <= len(v) <= 200}
        sorted_fields = sorted(valid.items(), key=lambda x: len(x[1]))

        self._lib_filter_fields = {
            _FILTER_FIELD_DISPLAY.get(f, f): f for f, _ in sorted_fields
        }
        display_names = list(self._lib_filter_fields.keys())

        # Build pre-selection order: preferred names first, then discovery order
        preferred = _TIR_DEFAULT_FILTERS if self._mode == 'LWIR' else _VNIR_DEFAULT_FILTERS
        seen: set[str] = set()
        ordered: list[str] = []
        for n in preferred:
            if n in self._lib_filter_fields and n not in seen:
                ordered.append(n)
                seen.add(n)
        for n in display_names:
            if n not in seen:
                ordered.append(n)
                seen.add(n)

        for i in range(_N_FILTER_ROWS):
            if display_names:
                self._filter_key_combos[i].config(values=display_names, state='readonly')
            else:
                self._filter_key_combos[i].config(values=[], state='disabled')

            if i < len(ordered):
                self._filter_key_vars[i].set(ordered[i])
                self._populate_val_combo(i)
            else:
                self._filter_key_vars[i].set('')
                self._filter_val_vars[i].set(ALL)
                self._filter_val_combos[i].config(values=[ALL], state='disabled')

    def _on_filter_key_changed(self, idx: int) -> None:
        self._filter_val_vars[idx].set(ALL)
        self._populate_val_combo(idx)
        self._apply_filters()

    def _populate_val_combo(self, idx: int) -> None:
        key_display = self._filter_key_vars[idx].get()
        field = self._lib_filter_fields.get(key_display, '')
        if not field:
            self._filter_val_combos[idx].config(values=[ALL], state='disabled')
            return
        _suppress = {'nan', '-', '0', '', 'none', 'n/a'}
        values = sorted({
            str(e.get(field, '')).strip()
            for e in self._browse_source.values()
            if str(e.get(field, '')).strip().lower() not in _suppress
        })
        self._filter_val_combos[idx].config(values=[ALL] + values, state='readonly')
        self._filter_val_vars[idx].set(ALL)

    # -----------------------------------------------------------------------
    # Filtering / listbox population
    # -----------------------------------------------------------------------

    def _format_entry_label(self, sid, entry: dict) -> str:
        """Format the listbox display string for a spectrum entry."""
        if self._mode == 'LWIR':
            spec_id = entry.get('spec_id', sid)
            try:
                return f"{int(spec_id):>6}  {str(entry.get('sample_name', ''))}"
            except (ValueError, TypeError):
                return f"{str(spec_id):>6}  {str(entry.get('sample_name', ''))}"
        else:
            return str(entry.get('label') or entry.get('sample_name') or sid)

    def _apply_filters(self) -> None:
        filtered = self._browse_source
        for i in range(_N_FILTER_ROWS):
            key_display = self._filter_key_vars[i].get()
            if not key_display:
                continue
            field = self._lib_filter_fields.get(key_display, '')
            if not field:
                continue
            sel = self._filter_val_vars[i].get()
            if sel == ALL or sel == '':
                continue
            filtered = {
                sid: e for sid, e in filtered.items()
                if str(e.get(field, '')).strip() == sel
            }

        self._filtered_ids = list(filtered.keys())
        self._avail_lb.delete(0, tk.END)
        for sid in self._filtered_ids:
            e = self._browse_source[sid]
            self._avail_lb.insert(tk.END, self._format_entry_label(sid, e))
        self._n_avail_label.config(text=f'{len(self._filtered_ids)} spectra')

    def _get_selected_ids(self) -> list:
        return [self._filtered_ids[i] for i in self._avail_lb.curselection()]

    # -----------------------------------------------------------------------
    # Active tab — drawing and info
    # -----------------------------------------------------------------------

    def _on_avail_selection_changed(self, _event) -> None:
        sel = self._avail_lb.curselection()
        if not sel:
            return
        if self._active_mode_var.get() == 'stacked' and len(sel) > 1:
            sids = [self._filtered_ids[i] for i in sel]
            self._draw_active_stacked(sids)
        else:
            self._draw_active(self._filtered_ids[sel[-1]])

    def _on_active_mode_change(self) -> None:
        """Redraw active figure when the Single/Stacked toggle changes."""
        sel = self._avail_lb.curselection()
        if not sel:
            return
        if self._active_mode_var.get() == 'stacked' and len(sel) > 1:
            sids = [self._filtered_ids[i] for i in sel]
            self._draw_active_stacked(sids)
        else:
            self._draw_active(self._filtered_ids[sel[-1]])

    def _draw_active(self, sid: int) -> None:
        """
        Draw the single active spectrum on the Active tab figure and populate
        the metadata treeview.  Looks up from _current_album first (preferred),
        falling back to _browse_source.
        """
        self._active_sid = sid

        # Remove previous secondary axis before cla()
        if self._active_secax is not None:
            try:
                self._active_secax.remove()
            except Exception:
                pass
            self._active_secax = None

        self._active_ax.cla()
        self._active_ax.set_xlabel(self._current_xlabel())
        self._active_ax.set_ylabel(self._ylabel())

        src = self._current_album if sid in self._current_album else self._browse_source
        if sid not in src:
            self._active_canvas.draw_idle()
            return

        entry = src[sid]
        x, y  = self._get_plot_xy(entry)

        self._active_ax.plot(x, y, color='tab:red', linewidth=1.0)

        xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
        ymin = max(0.0,  float(np.nanmin(y)) - 0.02)
        ymax = min(1.05, float(np.nanmax(y)) + 0.02)
        if self._x_ascending():
            self._active_ax.set_xlim(xmin, xmax)
        else:
            self._active_ax.set_xlim(xmax, xmin)
        self._active_ax.set_ylim(ymin, ymax)

        self._active_secax = self._add_top_axis(self._active_ax)
        self._active_canvas.draw_idle()

        self._update_info(sid, src)

    def _draw_active_stacked(self, sids: list) -> None:
        """
        Draw multiple selected spectra overlaid on the Active tab figure.

        Uses the default prop_cycle for colors.  A legend is shown when
        there are 15 or fewer spectra.  The info treeview reflects the
        last entry in *sids*.
        """
        if self._active_secax is not None:
            try:
                self._active_secax.remove()
            except Exception:
                pass
            self._active_secax = None

        self._active_ax.cla()
        self._active_ax.set_xlabel(self._current_xlabel())
        self._active_ax.set_ylabel(self._ylabel())

        x_all, y_all = [], []
        last_sid = sids[-1]

        for sid in sids:
            src   = self._current_album if sid in self._current_album else self._browse_source
            if sid not in src:
                continue
            entry = src[sid]
            x, y  = self._get_plot_xy(entry)
            lbl   = self._format_entry_label(sid, entry)
            self._active_ax.plot(x, y, linewidth=1.0, label=lbl)
            x_all.append(x)
            y_all.append(y)

        if x_all:
            xs = np.concatenate(x_all)
            ys = np.concatenate(y_all)
            xmin, xmax = float(np.nanmin(xs)), float(np.nanmax(xs))
            ymin = max(0.0,  float(np.nanmin(ys)) - 0.02)
            ymax = min(1.05, float(np.nanmax(ys)) + 0.02)
            if self._x_ascending():
                self._active_ax.set_xlim(xmin, xmax)
            else:
                self._active_ax.set_xlim(xmax, xmin)
            self._active_ax.set_ylim(ymin, ymax)

        if len(sids) <= 15:
            self._active_ax.legend(
                fontsize=7, loc='best',
                framealpha=0.7, handlelength=1.2, handletextpad=0.4,
            )

        self._active_secax = self._add_top_axis(self._active_ax)
        self._active_canvas.draw_idle()

        # Info treeview: show metadata for the last selected spectrum
        src = self._current_album if last_sid in self._current_album else self._browse_source
        self._update_info(last_sid, src)

    def _update_info(self, sid, src: dict) -> None:
        if sid not in src:
            return
        entry = src[sid]

        for row in self._info_tree.get_children():
            self._info_tree.delete(row)

        _suppress = {'nan', '', '-', 'none', 'n/a'}
        info_groups = _INFO_GROUPS if self._mode == 'LWIR' else _VNIR_INFO_GROUPS

        for group_name, fields in info_groups:
            collapsed = group_name in _INFO_COLLAPSED_BY_DEFAULT
            grp_iid   = self._info_tree.insert(
                '', tk.END, text='', values=(group_name, ''),
                open=not collapsed, tags=('header',),
            )
            row_idx = 0
            for field, label in fields:
                v = entry.get(field, '')
                if hasattr(v, 'item'):
                    v = v.item()
                if str(v).strip().lower() in _suppress:
                    continue
                tag = 'odd' if row_idx % 2 else 'even'
                self._info_tree.insert(
                    grp_iid, tk.END, text='', values=(label, v), tags=(tag,))
                row_idx += 1

    def _rebuild_summary_tree(self, fields: list[tuple[str, str]]) -> None:
        """Destroy and recreate the summary Treeview with a new column layout."""
        if self._summary_frame is not None:
            self._summary_frame.destroy()

        frame = ttk.Frame(self._summary_parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self._summary_frame = frame

        col_keys = [f for f, _ in fields]
        vs = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        hs = ttk.Scrollbar(frame, orient=tk.HORIZONTAL)
        self._summary_tree = ttk.Treeview(
            frame, columns=col_keys,
            show='headings',
            yscrollcommand=vs.set,
            xscrollcommand=hs.set,
            selectmode='none',
        )
        vs.config(command=self._summary_tree.yview)
        hs.config(command=self._summary_tree.xview)

        for hdf_key, label in fields:
            self._summary_tree.heading(hdf_key, text=label)
            if hdf_key in ('spec_id',):
                self._summary_tree.column(hdf_key, width=60,  stretch=False, anchor=tk.CENTER)
            elif hdf_key in ('sample_name', 'label', 'source'):
                self._summary_tree.column(hdf_key, width=200, stretch=True)
            else:
                self._summary_tree.column(hdf_key, width=110, stretch=False)

        self._summary_tree.grid(row=0, column=0, sticky=tk.NSEW)
        vs.grid(row=0, column=1, sticky=tk.NS)
        hs.grid(row=1, column=0, sticky=tk.EW)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        self._summary_tree.tag_configure('odd',  background='#f5f5f5')
        self._summary_tree.tag_configure('even', background='#ffffff')

    def _update_summary(self) -> None:
        """Rebuild the Summary tab Treeview from the current album."""
        fields = _SUMMARY_FIELDS if self._mode == 'LWIR' else _VNIR_SUMMARY_FIELDS
        self._rebuild_summary_tree(fields)

        if not self._current_album:
            return

        records = [
            {hdf_key: entry.get(hdf_key, '') for hdf_key, _ in fields}
            for entry in self._current_album.values()
        ]
        df = pd.DataFrame(records)

        _suppress = {'nan', 'None', ''}
        for i, (_, row) in enumerate(df.iterrows()):
            tag = 'odd' if i % 2 else 'even'
            values = [
                str(row[hdf_key]) if str(row[hdf_key]) not in _suppress else '\u2014'
                for hdf_key, _ in fields
            ]
            self._summary_tree.insert('', tk.END, values=values, tags=(tag,))

    # -----------------------------------------------------------------------
    # Sampling helpers
    # -----------------------------------------------------------------------

    def _get_plot_xy(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (x, y) for plotting, applying the current sampling if set.
        Inserts NaN break-points at large gaps in x (e.g. TES73 atmospheric
        window) so matplotlib renders them as gaps rather than spanning lines.
        The stored entry data is never modified.
        """
        opt = self._sampling_opts[self._current_samp]
        if opt['xaxis'] is None:
            x, y = entry['xaxis'], entry['data']
        else:
            y = resample_spectrum(entry['xaxis'], entry['data'],
                                  opt['xaxis'], opt['is_wl'])
            x = opt['xaxis']
        return insert_plot_gaps(x, y)

    def _current_xlabel(self) -> str:
        if self._mode == 'VSWIR':
            return 'Wavelength (nm)'
        return self._sampling_opts[self._current_samp]['xlabel']

    def _ylabel(self) -> str:
        return 'Reflectance' if self._mode == 'VSWIR' else 'Emissivity'

    def _x_is_wl(self) -> bool:
        """True when the current x-axis is wavelength (µm) — TIR instrument grids only."""
        return self._sampling_opts[self._current_samp]['is_wl']

    def _x_ascending(self) -> bool:
        """True when x should be plotted low→high (VNIR nm, or TIR wavelength grids)."""
        return self._mode == 'VSWIR' or self._x_is_wl()

    def _default_xlim(self) -> tuple[float, float]:
        """Return (lo, hi) for the empty-album state — already in display order."""
        if self._mode == 'VSWIR':
            return 300.0, 2500.0
        return 4000.0, 200.0  # wavenumber: plotted high→low

    # -----------------------------------------------------------------------
    # Album plot drawing
    # -----------------------------------------------------------------------

    def _draw_album(self) -> None:
        """Central draw routine for the Album tab — clears and redraws according to mode."""
        # Remove previous secondary axis before cla() to avoid stale handles.
        if self._secax is not None:
            try:
                self._secax.remove()
            except Exception:
                pass
            self._secax = None

        self._ax.cla()
        self._ax.set_xlabel(self._current_xlabel())
        self._ax.set_ylabel(self._ylabel())

        ids = list(self._current_album.keys())
        lo, hi = self._default_xlim()
        if not ids:
            self._ax.set_xlim(lo, hi)
            self._ax.set_ylim(0, 1.05)
            self._update_nav_label()
        elif self._plot_mode == 'stacked':
            self._draw_stacked(ids)
        else:
            self._draw_individual(ids)

        self._secax = self._add_top_axis(self._ax)
        self._canvas.draw_idle()

    def _draw_stacked(self, ids: list[int]) -> None:
        n_pages = max(1, (len(ids) + MAX_STACKED - 1) // MAX_STACKED)
        self._stacked_page = max(0, min(self._stacked_page, n_pages - 1))

        start = self._stacked_page * MAX_STACKED
        shown = ids[start : start + MAX_STACKED]

        for i, sid in enumerate(shown):
            entry = self._current_album[sid]
            x, y  = self._get_plot_xy(entry)
            color = _COLOR_CYCLE[i % len(_COLOR_CYCLE)]
            lbl   = str(entry.get('label') or entry.get('sample_name') or sid)
            self._ax.plot(x, y, color=color, linewidth=0.9, label=lbl)

        if shown:
            self._set_axis_limits(shown)
            if self._mode == 'LWIR':
                self._draw_atmosphere_spans()
            self._ax.legend(fontsize=7, loc='lower right',
                            framealpha=0.7, ncol=1)

        end_label = min(start + MAX_STACKED, len(ids))
        self._update_nav_label(
            current=f'{start + 1}–{end_label}',
            total=len(ids),
        )

    def _draw_individual(self, ids: list[int]) -> None:
        self._indiv_idx = max(0, min(self._indiv_idx, len(ids) - 1))
        sid   = ids[self._indiv_idx]
        entry = self._current_album[sid]
        x, y  = self._get_plot_xy(entry)

        lbl = str(entry.get('label') or entry.get('sample_name') or sid)
        self._ax.plot(x, y, color='tab:red', linewidth=1.0, label=lbl)
        self._set_axis_limits([sid])
        if self._mode == 'LWIR':
            self._draw_atmosphere_spans()
        self._ax.legend(fontsize=9, loc='lower right', framealpha=0.7)

        self._update_nav_label(self._indiv_idx + 1, len(ids))

        # Sync Active tab to the currently displayed spectrum
        self._draw_active(sid)

    def _set_axis_limits(self, ids: list) -> None:
        all_x, all_y = [], []
        for sid in ids:
            x, y = self._get_plot_xy(self._current_album[sid])
            all_x.append(x)
            all_y.append(y)

        ax_x = np.concatenate(all_x)
        ax_y = np.concatenate(all_y)
        xmin = float(np.nanmin(ax_x))
        xmax = float(np.nanmax(ax_x))
        ymin = max(0.0,  float(np.nanmin(ax_y)) - 0.02)
        ymax = min(1.05, float(np.nanmax(ax_y)) + 0.02)

        if self._x_ascending():
            self._ax.set_xlim(xmin, xmax)
        else:
            self._ax.set_xlim(xmax, xmin)

        self._ax.set_ylim(ymin, ymax)

    def _draw_atmosphere_spans(self) -> None:
        """Overlay atmospheric opacity/window shading on the current axes."""
        spans = _ATMOSPHERE_SPANS[self._atm_var.get()]
        seen_labels: set[str] = set()
        for wn_lo, wn_hi, color, label in spans:
            if self._x_is_wl():
                lo = 1e4 / wn_hi if wn_hi > 0 else 0.0
                hi = 1e4 / wn_lo if wn_lo > 0 else 1e6
            else:
                lo, hi = wn_lo, wn_hi
            legend_label = label if label not in seen_labels else '_nolegend_'
            seen_labels.add(label)
            self._ax.axvspan(lo, hi, color=color, alpha=0.3,
                             zorder=0, label=legend_label)

    def _add_top_axis(self, ax) -> object:
        """
        Add a secondary x-axis on top showing the complementary unit.

        TIR primary cm\u207b\xb9 \u2192 secondary \u00b5m (1e4/x, self-inverse).
        VNIR primary nm  \u2192 secondary \u00b5m (x/1000, inverse x*1000).

        Returns the secondary axis object, or None if limits are invalid.
        """
        lo, hi = ax.get_xlim()
        if not (lo > 0 and hi > 0):
            return None

        if self._mode == 'VSWIR':
            def _fw(x: np.ndarray) -> np.ndarray:
                with np.errstate(divide='ignore'):
                    return np.where(np.asarray(x) != 0,
                                    np.asarray(x, dtype=float) / 1000.0, np.nan)
            def _bw(x: np.ndarray) -> np.ndarray:
                with np.errstate(divide='ignore'):
                    return np.where(np.asarray(x) != 0,
                                    np.asarray(x, dtype=float) * 1000.0, np.nan)
            secax = ax.secondary_xaxis('top', functions=(_fw, _bw))
            secax.set_xlabel('Wavelength (\u03bcm)')
            return secax

        # TIR: wavenumber cm\u207b\xb9 \u2194 \u00b5m (self-inverse: 1e4/x)
        def _fw(x: np.ndarray) -> np.ndarray:
            with np.errstate(divide='ignore'):
                return np.where(np.asarray(x) != 0, 1e4 / np.asarray(x, dtype=float), np.nan)

        secax = ax.secondary_xaxis('top', functions=(_fw, _fw))
        if self._x_is_wl():
            secax.xaxis.set_ticks([500, 1000, 2000, 3000, 4000, 5000])
            secax.set_xlabel('Wavenumber (cm\u207b\xb9)')
        else:
            secax.xaxis.set_ticks(
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50])
            secax.set_xlabel('Wavelength (\u03bcm)')
        return secax

    def _update_nav_label(self, current: int | str = 0, total: int = 0) -> None:
        if total:
            self._nav_label.config(text=f'{current} / {total}')
        else:
            self._nav_label.config(text='— / —')

    # -----------------------------------------------------------------------
    # Mode / sampling controls
    # -----------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        self._plot_mode = self._mode_var.get()
        self._stacked_page = 0
        self._indiv_idx    = 0
        self._draw_album()

    def _on_sampling_change(self) -> None:
        self._current_samp = self._samp_var.get()
        self._draw_album()
        # Redraw active if one is selected
        if self._active_sid is not None:
            self._draw_active(self._active_sid)

    def _on_prev(self) -> None:
        if not self._current_album:
            return
        ids = list(self._current_album.keys())
        if self._plot_mode == 'stacked':
            n_pages = max(1, (len(ids) + MAX_STACKED - 1) // MAX_STACKED)
            self._stacked_page = (self._stacked_page - 1) % n_pages
        else:
            self._indiv_idx = (self._indiv_idx - 1) % len(ids)
        self._draw_album()

    def _on_next(self) -> None:
        if not self._current_album:
            return
        ids = list(self._current_album.keys())
        if self._plot_mode == 'stacked':
            n_pages = max(1, (len(ids) + MAX_STACKED - 1) // MAX_STACKED)
            self._stacked_page = (self._stacked_page + 1) % n_pages
        else:
            self._indiv_idx = (self._indiv_idx + 1) % len(ids)
        self._draw_album()

    # -----------------------------------------------------------------------
    # Album actions
    # -----------------------------------------------------------------------

    def _on_add_to_album(self) -> None:
        added = 0
        for sid in self._get_selected_ids():
            if sid not in self._current_album:
                self._current_album[sid] = self._browse_source[sid]
                e = self._browse_source[sid]
                self._album_lb.insert(
                    tk.END,
                    self._format_entry_label(sid, e),
                )
                added += 1
        if added:
            self._n_album_label.config(text=f'{len(self._current_album)} spectra')
            self._draw_album()
            self._update_summary()
        if self._browse_name == _SRC_ALBUM:
            self._update_status()
            self._apply_filters()

    def _on_album_remove(self) -> None:
        selected  = sorted(self._album_lb.curselection(), reverse=True)
        album_ids = list(self._current_album.keys())
        for i in selected:
            sid = album_ids[i]
            del self._current_album[sid]
            self._album_lb.delete(i)
            # Clear active display if the removed spectrum was active
            if self._active_sid == sid:
                self._active_sid = None
                self._active_ax.cla()
                self._active_ax.set_xlabel(self._current_xlabel())
                self._active_ax.set_ylabel(self._ylabel())
                lo, hi = self._default_xlim()
                self._active_ax.set_xlim(lo, hi)
                self._active_ax.set_ylim(0, 1.05)
                self._active_canvas.draw_idle()
                for row in self._info_tree.get_children():
                    self._info_tree.delete(row)

        self._n_album_label.config(text=f'{len(self._current_album)} spectra')
        self._draw_album()
        self._update_summary()
        if self._browse_name == _SRC_ALBUM:
            self._update_status()
            self._apply_filters()

    def _on_reset_album(self) -> None:
        if self._current_album:
            if not messagebox.askyesno('Reset Album',
                                       'Clear all spectra from the current album?'):
                return
        self._current_album.clear()
        self._stacked_page = 0
        self._indiv_idx    = 0
        self._album_lb.delete(0, tk.END)
        self._n_album_label.config(text='0 spectra')

        # Clear active display
        self._active_sid = None
        self._active_ax.cla()
        self._active_ax.set_xlabel(self._current_xlabel())
        self._active_ax.set_ylabel(self._ylabel())
        lo, hi = self._default_xlim()
        self._active_ax.set_xlim(lo, hi)
        self._active_ax.set_ylim(0, 1.05)
        self._active_canvas.draw_idle()
        for row in self._info_tree.get_children():
            self._info_tree.delete(row)

        self._draw_album()
        self._update_summary()
        if self._browse_name == _SRC_ALBUM:
            self._update_status()
            self._apply_filters()

    # -----------------------------------------------------------------------
    # Toolbar actions
    # -----------------------------------------------------------------------

    def _on_source_changed(self) -> None:
        self._set_browse_source(self._src_var.get())

    def _on_load(self) -> None:
        lib_dir = get_config().get('spectral_libraries_dir') or '/'
        path = filedialog.askopenfilename(
            title='Load spectral library',
            initialdir=lib_dir,
            filetypes=[('HDF5 files', '*.hdf *.h5 *.hdf5'), ('All files', '*.*')],
        )
        if not path:
            return
        stem = Path(path).stem
        name = stem
        idx  = 2
        while name in self._extra_libs or name in (_SRC_FULL, _SRC_ALBUM):
            name = f'{stem} ({idx})'
            idx += 1
        self._extra_libs[name] = path
        self._update_src_combo()
        self._src_var.set(name)
        self._set_browse_source(name)

    def _on_export(self) -> None:
        if not self._current_album:
            messagebox.showinfo('Export', 'Current album is empty — nothing to export.')
            return
        default_name = f'speclib_{datetime.now().strftime("%Y%m%d_%H%M%S")}.hdf'
        dlg = ExportDialog(self, self._sampling_opts, default_name, mode=self._mode)
        if dlg.result is None:
            return
        path     = dlg.result['filename']
        sampling = dlg.result['sampling']

        if self._mode == 'VSWIR':
            vnir_opt = _VNIR_SAMPLING_OPTS[sampling]
            target   = vnir_opt['xaxis']
            if target is not None:
                album: dict = {}
                for sid, entry in self._current_album.items():
                    resampled          = dict(entry)
                    src_x              = np.asarray(entry['xaxis'], dtype=float)
                    src_d              = np.asarray(entry['data'],  dtype=float)
                    # ensure ascending before interp
                    if src_x[-1] < src_x[0]:
                        src_x = src_x[::-1]
                        src_d = src_d[::-1]
                    resampled['data']  = np.interp(target, src_x, src_d,
                                                   left=np.nan, right=np.nan)
                    resampled['xaxis'] = target
                    album[sid] = resampled
            else:
                album = self._current_album
        else:
            opt = self._sampling_opts[sampling]
            if opt['xaxis'] is not None:
                album = {}
                for sid, entry in self._current_album.items():
                    resampled          = dict(entry)
                    resampled['data']  = resample_spectrum(
                        entry['xaxis'], entry['data'], opt['xaxis'], opt['is_wl'])
                    resampled['xaxis'] = opt['xaxis']
                    album[sid] = resampled
            else:
                album = self._current_album

        try:
            _export_album(album, path, mode=self._mode)
            messagebox.showinfo('Export',
                                f'Exported {len(album)} spectra to:\n{path}')
        except Exception as exc:
            messagebox.showerror('Export error', str(exc))


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

class ExportDialog(tk.Toplevel):
    """
    Modal dialog for album export options.

    Parameters
    ----------
    parent : tk.Widget
    sampling_opts : dict
        TIR instrument presets {display_name: {xaxis, is_wl, xlabel}}.
    default_filename : str
        Pre-filled filename suggestion (basename only; browse updates the full path).
    mode : str
        'LWIR' or 'VSWIR'. Controls which resolution options are shown.

    Attributes
    ----------
    result : dict | None
        {'filename': str, 'sampling': str} if OK was clicked, else None.
    """

    def __init__(self, parent: tk.Widget, sampling_opts: dict,
                 default_filename: str, *, mode: str = 'LWIR') -> None:
        super().__init__(parent)
        self.title('Export Options')
        self.resizable(False, False)
        self.grab_set()
        self.result: dict | None = None

        if mode == 'VSWIR':
            choices = list(_VNIR_SAMPLING_OPTS.keys())
            label   = 'Resolution:'
        else:
            choices = list(sampling_opts.keys())
            label   = 'Sampling:'

        # Filename row
        ttk.Label(self, text='Filename:').grid(
            row=0, column=0, sticky=tk.W, padx=8, pady=(10, 4))
        self._fname_var = tk.StringVar(value=default_filename)
        ttk.Entry(self, textvariable=self._fname_var, width=42).grid(
            row=0, column=1, padx=(0, 4), pady=(10, 4), sticky=tk.EW)
        ttk.Button(self, text='Browse…', command=self._browse).grid(
            row=0, column=2, padx=(0, 8), pady=(10, 4))

        # Sampling / Resolution row
        ttk.Label(self, text=label).grid(
            row=1, column=0, sticky=tk.W, padx=8, pady=4)
        self._samp_var = tk.StringVar(value=choices[0])
        ttk.Combobox(
            self, textvariable=self._samp_var,
            state='readonly', width=46,
            values=choices,
        ).grid(row=1, column=1, columnspan=2, padx=(0, 8), pady=4, sticky=tk.EW)

        # OK / Cancel
        btn_row = ttk.Frame(self)
        btn_row.grid(row=2, column=0, columnspan=3, pady=(8, 10))
        ttk.Button(btn_row, text='OK',     command=self._ok    ).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_row, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=8)

        self.columnconfigure(1, weight=1)
        self.protocol('WM_DELETE_WINDOW', self.destroy)
        self.transient(parent)
        self.wait_window()

    def _browse(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self,
            title='Export album as HDF5',
            initialfile=Path(self._fname_var.get()).name,
            defaultextension='.hdf',
            filetypes=[('HDF5 files', '*.hdf *.h5 *.hdf5'), ('All files', '*.*')],
        )
        if path:
            self._fname_var.set(path)

    def _ok(self) -> None:
        fname = self._fname_var.get().strip()
        if not fname:
            messagebox.showwarning('Export', 'Please enter a filename.', parent=self)
            return
        self.result = {'filename': fname, 'sampling': self._samp_var.get()}
        self.destroy()


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------

def _export_album(album: dict, path: str, *, mode: str = 'LWIR') -> None:
    """
    Write album spectra to HDF5 using saveDVhdf.

    Each spectrum is stored as a group keyed by its spec_id (as a string).
    The internal ``hdf_group`` bookkeeping key is excluded from the output.
    ``x_unit`` / ``y_unit`` are injected (or overwritten) so the exported
    file is self-describing for :func:`_detect_library_mode`.
    The resulting file is readable by ``_load_hdf`` / ``readDVhdf``.
    """
    x_unit = 'nm'          if mode == 'VSWIR' else 'cm-1'
    y_unit = 'reflectance' if mode == 'VSWIR' else 'emissivity'
    _SKIP  = frozenset({'hdf_group'})
    data = {
        str(sid): {
            **{k: v for k, v in entry.items() if k not in _SKIP},
            'x_unit': x_unit,
            'y_unit': y_unit,
        }
        for sid, entry in album.items()
    }
    saveDVhdf(data, path)


# ---------------------------------------------------------------------------

def launch(mode: str = 'LWIR', library: str | None = None) -> None:
    app = SpeclibViewer(default_mode=mode, library_path=library)
    app._standalone_root.mainloop()


if __name__ == '__main__':
    launch()
