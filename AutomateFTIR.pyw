#!/usr/bin/env python3
"""
AutomateFTIR — GUI for automated FTIR spectroscopy data collection.

Supports three measurement modes:
  • Emission     — heated-sample emission with dual blackbody calibration
  • Transmission — sample transmittance relative to a background spectrum
  • Reflectance  — diffuse/specular reflectance relative to a background spectrum

Layout
------
Top     : mode selector | OMNIC parameter file | collection buttons | save folder
Main    : matplotlib spectral display (wavenumber bottom / wavelength top)
          | right panel with live multimeter readings

Hardware
--------
Spectrometer : Thermo Nicolet FTIR controlled via OMNIC DDE interface (Windows).
Multimeter   : Keithley 2700 via PyVISA over TCP/IP (TCPIP::…::SOCKET).
              Used for temperature monitoring in Emission mode; display-only in T/R.
"""

# Standard library
import csv
import logging
import logging.handlers
import subprocess
import threading
import time

# Numerics
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Callable

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Hardware — Windows only
import pyvisa
import win32ui  # initialises the win32 OLE layer required by dde  # noqa: F401
import dde

# ---------------------------------------------------------------------------
# Package bootstrap — allow running directly as a script
# ---------------------------------------------------------------------------
if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'speclab'

from .plot import _add_top_axis
from .utils import readOMNIC, normalize, r2t_nau, c2k
from .functions import emcal, tracal, refcal, MissingTempsError
from . import __version__

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.size':             11,
    'lines.linewidth':       1.0,
    'xtick.direction':       'in',
    'xtick.top':             True,
    'xtick.labelsize':       11,
    'ytick.direction':       'in',
    'ytick.right':           True,
    'ytick.labelsize':       11,
    'axes.grid':             True,
    'axes.axisbelow':        True,
    'axes.labelsize':        12,
    'axes.titlesize':        12,
    'grid.linestyle':        '--',
    'axes.formatter.limits': (-4, 4),
    'errorbar.capsize':      2,
})

# Suppress PyVISA's serial-interface registration warnings (PySerial not
# installed; TCP/IP is the only transport used here).
logging.getLogger('pyvisa').setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Color schemes
# ---------------------------------------------------------------------------

def _lighten_colors(colors: list, factor: float = 0.5) -> list:
    import colorsys
    out = []
    for r, g, b, a in colors:
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l2 = l + (1.0 - l) * factor
        r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s)
        out.append((r2, g2, b2, a))
    return out

_tab20_dark  = [plt.cm.tab20(i) for i in range(0, 20, 2)]
_tab20_light = [plt.cm.tab20(i) for i in range(1, 20, 2)]
_dark2       = [(*c, 1.0) for c in plt.cm.Dark2.colors]

_COLOR_SCHEMES: dict[str, list] = {
    'Standard': _tab20_dark + _tab20_light,            # 20 — tab20 dark+light hues
    'Dark':     _dark2 + _lighten_colors(_dark2),      # 16 — high-contrast Dark2
}

try:
    import cmcrameri as _cmcrameri
    for _cmc_name in ['batlowS', 'hawaiiS', 'lipariS', 'tokyoS']:
        try:
            _COLOR_SCHEMES[_cmc_name] = list(getattr(_cmcrameri.cm, _cmc_name).colors)
        except AttributeError:
            pass
except ImportError:
    pass

_COLOR_SCHEME_DEFAULT = 'Dark'

# Fixed colors for blackbody spectra (not drawn from the sample color cycle).
_BB_HOT_COLOR  = 'red'
_BB_WARM_COLOR = 'blue'

# Vibrant color cycler for T/R background and blank spectra.
# Drawn from matplotlib's Set1 (9 highly saturated hues), distinct from the
# sample cyclers so backgrounds and blanks are instantly recognisable.
_VIBRANT_BKG_COLORS = [plt.cm.Set1(i) for i in range(9)]

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------

MULTIMETER_ADDRESS   = 'TCPIP::10.12.100.246::1394::SOCKET'
OMNIC_SERVER_NAME    = 'OMNIC'
OMNIC_TOPIC_NAME     = 'SPECTRA'
OMNIC_PARAM_DIR      = Path(r'C:\my documents\omnic\Param')
OMNIC_EXE            = Path(r'C:\Program Files (x86)\omnic\omnic32.exe')
DEFAULT_DATA_DIR     = Path(r'C:\Users\ftir_spec_user\Nextcloud\107 Storage\FTIR Data')
DEFAULT_EXP_FILENAME = 'Emission_MIR - Heated Sample.exp'  # legacy alias

# Per-mode OMNIC experiment parameter file defaults and keyword filters.
_MODE_EXP_DEFAULTS: dict[str, str] = {
    'Emission':     'Emission_MIR - Heated Sample.exp',
    'Transmission': 'Transmission MIR.exp',
    'Reflectance':  'DiffusIR_MIR.exp',
}
_MODE_EXP_KEYWORDS: dict[str, list[str]] = {
    'Emission':     ['emission'],
    'Transmission': ['transmiss'],
    'Reflectance':  ['diffusir', 'reflect'],
}

# Auto-connect polling for OMNIC (ms between attempts, total timeout in seconds).
OMNIC_AUTOCONNECT_POLL_MS  = 5_000
OMNIC_AUTOCONNECT_TIMEOUT_S = 120

# Polling interval for live multimeter display (seconds).
MM_POLL_INTERVAL_S = 60

# Purge equilibration delay shown to the user before every T/R collection (s).
PURGE_DELAY_S = 30

# Retry parameters for CollectSample DDE command (OMNIC may NACK when busy).
COLLECT_MAX_RETRIES   = 30
COLLECT_RETRY_DELAY_S = 2

# Multimeter sampling interval during a spectrometer collection (seconds).
MM_MEASUREMENT_INTERVAL_S = 10

# BB auto-selection thresholds (°C).  When the dialog opens with a live
# temperature reading, the type is pre-selected based on these values.
_BB_TEMP_WARM_MIN_C = 40.0   # below this: no auto-selection (BB not ready)
_BB_TEMP_HOT_MIN_C  = 90.0   # at or above this: auto-select BB Hot

# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------

_CHANNEL_LABELS: dict[int, str] = {
    101: '101: PRT Resistance',
    102: '102: PRT Resistance',
    103: '103: Mirror',
    104: '104: Chamber exterior',
    105: '105: Chamber interior',
    106: '106: Chamber door',
    107: '107: Detector',
}

_CHANNEL_UNITS: dict[int, str] = {
    101: 'Ω',
    102: 'Ω',
    103: '°C',
    104: '°C',
    105: '°C',
    106: '°C',
    107: '°C',
}

# Display order: ascending channels; separator falls between resistance and temperature.
_PANEL_ORDER: list[int] = [101, 102, 103, 104, 105, 106, 107]

# Channel after which a visual separator is inserted in the multimeter panel.
_PANEL_SEPARATOR_AFTER: int = 102


# ---------------------------------------------------------------------------
# Indicator light widget
# ---------------------------------------------------------------------------

class IndicatorLight(tk.Canvas):
    """
    Small circular LED-style connection indicator.

    States
    ------
    'ok'    : green  — instrument connected and responding
    'error' : red    — connection lost or error
    'idle'  : gray   — not yet connected (default)
    """

    _COLORS: dict[str, str] = {
        'ok':    '#22bb44',
        'error': '#cc3333',
        'idle':  '#999999',
    }

    def __init__(self, parent: tk.Misc, size: int = 14, **kwargs) -> None:
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, **kwargs)
        r = size // 2 - 1
        c = size // 2
        self._oval = self.create_oval(
            c - r, c - r, c + r, c + r,
            fill=self._COLORS['idle'], outline='',
        )

    def set_state(self, state: str) -> None:
        """
        Update the indicator colour.

        Parameters
        ----------
        state : str
            One of ``'ok'``, ``'error'``, or ``'idle'``.
        """
        self.itemconfig(self._oval,
                        fill=self._COLORS.get(state, self._COLORS['idle']))


# ---------------------------------------------------------------------------
# EmcalOptionsDialog
# ---------------------------------------------------------------------------

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
            ('lab',              'Lab',                          'combo', ['nau', 'asu', 'swri', 'spectrometer']),
            ('method',           'Method',                       'combo', ['nem', 'alpha', 'hullfit_linear', 'hullfit', 'mmd']),
            ('max_emiss',        'Max emissivity',               'float', None),
            ('bb_emiss',         'BB emissivity',                'float', None),
            ('n_bb',             'N BB (hullfit)',               'int',   None),
            ('temp_halfwidth',   'Temp half-width K (hullfit)',  'float', None),
            ('violation_weight', 'Violation weight (hullfit)',   'float', None),
            ('violation_tol',    'Violation tol (hullfit)',      'float', None),
            ('escalation_factor','Escalation factor (hullfit)',  'float', None),
            ('max_escalations',  'Max escalations (hullfit)',    'int',   None),
            ('noise_free',       'Noise-free IRF',               'bool',  None),
            ('apply_dehyd',      'Apply dehyd',                  'bool',  None),
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


# ---------------------------------------------------------------------------
# ExperimentParamsDialog
# ---------------------------------------------------------------------------

class ExperimentParamsDialog(tk.Toplevel):
    """Read-only display of the current OMNIC experiment parameters."""

    _LABELS: list[tuple[str, str]] = [
        ('exp_file',      'Parameter file'),
        ('resolution',    'Resolution (cm⁻¹)'),
        ('num_scans',     'Number of scans'),
        ('apodization',   'Apodization'),
        ('zero_fill',     'Zero fill'),
        ('high_cutoff',   'High cutoff (cm⁻¹)'),
        ('low_cutoff',    'Low cutoff (cm⁻¹)'),
        ('gain',          'Gain'),
        ('beamsplitter',  'Beamsplitter'),
        ('velocity',      'Velocity'),
    ]

    def __init__(self, master: tk.Misc, params: dict) -> None:
        super().__init__(master)
        self.title('Experiment Parameters')
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()
        frm = ttk.Frame(self, padding=14)
        frm.pack(fill=tk.BOTH, expand=True)
        for row, (key, label) in enumerate(self._LABELS):
            ttk.Label(frm, text=f'{label}:', anchor=tk.W).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 16), pady=2)
            ttk.Label(frm, text=params.get(key, '—'), anchor=tk.W,
                      font=('TkFixedFont', 10)).grid(
                row=row, column=1, sticky=tk.W, pady=2)
        ttk.Button(frm, text='Close', command=self.destroy).grid(
            row=len(self._LABELS), column=0, columnspan=2, pady=(12, 0))


# ---------------------------------------------------------------------------
# MultimeterController
# ---------------------------------------------------------------------------

class MultimeterController:
    """
    Manages the PyVISA connection and channel reads for the Keithley 2700.

    Thread safety
    -------------
    ``read_channel`` acquires ``_lock`` around each VISA query to prevent
    interleaving of writes from the live-poll thread and collection threads.

    Parameters
    ----------
    address : str
        VISA resource string (e.g. ``'TCPIP::10.12.100.246::1394::SOCKET'``).
    """

    # Channels 101–102 use 4-wire resistance; 103–107 use temperature.
    _CHANNEL_MODES: dict[int, str] = {
        101: 'FRES',
        102: 'FRES',
        103: 'Temperature',
        104: 'Temperature',
        105: 'Temperature',
        106: 'Temperature',
        107: 'Temperature',
    }

    # Trailing chars to strip from the SCPI response token.
    # Empirical from ftir-automation-v4.py: FRES strips 5, Temperature strips 2.
    _STRIP: dict[str, int] = {'FRES': 5, 'Temperature': 2}

    def __init__(self, address: str) -> None:
        self._address: str                     = address
        self._resource: pyvisa.Resource | None = None
        self._lock: threading.Lock             = threading.Lock()

    @property
    def connected(self) -> bool:
        """True if a VISA resource is currently open."""
        return self._resource is not None

    def connect(self) -> bool:
        """
        Open the VISA resource at the configured address.

        Returns
        -------
        bool
            True on success, False on any connection error.
        """
        self.disconnect()   # close any existing session before opening a new one
        try:
            rm  = pyvisa.ResourceManager()
            res = rm.open_resource(self._address)
            res.read_termination = '\n'
            self._resource = res
            logging.info("Multimeter connected: %s", self._address)
            return True
        except Exception as exc:
            logging.error("Multimeter connect failed: %s", exc)
            self._resource = None
            return False

    def disconnect(self) -> None:
        """Close the VISA resource if open."""
        if self._resource is not None:
            try:
                self._resource.close()
            except Exception:
                pass
            self._resource = None

    def read_channel(self, channel: int) -> float:
        """
        Read one Keithley channel.

        Closes the relay for *channel*, sets the appropriate measurement
        function, and queries a fresh reading via ``:SENSe:DATA:FRESh?``.

        Parameters
        ----------
        channel : int
            Channel number (101–107).

        Returns
        -------
        float
            Measured value in Ω for channels 101–102, °C for 103–107.

        Raises
        ------
        RuntimeError
            If the multimeter is not connected.
        KeyError
            If *channel* is not in the channel map.
        pyvisa.errors.VisaIOError
            On instrument communication failure.
        """
        if self._resource is None:
            raise RuntimeError("Multimeter not connected")
        mode  = self._CHANNEL_MODES[channel]
        strip = self._STRIP[mode]
        with self._lock:
            try:
                self._resource.write(f':ROUTe:CLOSe (@{channel})')
                self._resource.write(f":SENSe:FUNCtion '{mode}'")
                raw = self._resource.query(':SENSe:DATA:FRESh?')
            except pyvisa.errors.VisaIOError as exc:
                # TCP connection lost; tear down so the next connect() starts clean.
                self.disconnect()
                raise
        return float(raw.split(',')[0][:-strip])

    def read_all_channels(self, channels: list[int]) -> dict[int, float]:
        """
        Read multiple channels sequentially.

        Individual channel failures are logged and stored as ``float('nan')``
        so a partial result is always returned rather than raising.

        Parameters
        ----------
        channels : list of int
            Channel numbers to read.

        Returns
        -------
        dict[int, float]
            Channel number → measured value; failed channels contain NaN.
        """
        readings: dict[int, float] = {}
        for ch in channels:
            try:
                readings[ch] = self.read_channel(ch)
            except Exception as exc:
                logging.error("Channel %d read failed: %s", ch, exc)
                readings[ch] = float('nan')
        return readings


# ---------------------------------------------------------------------------
# SpectrometerController
# ---------------------------------------------------------------------------

class SpectrometerController:
    """
    Manages DDE communication with the OMNIC spectrometer application.

    OMNIC must be open and fully loaded before ``connect`` is called.
    All DDE commands execute synchronously; ``collect`` blocks its calling
    thread until the scan reaches 100 %.

    Parameters
    ----------
    server_name : str
        DDE server name (``'OMNIC'``).
    topic_name : str
        DDE topic (``'SPECTRA'``).
    """

    def __init__(self, server_name: str, topic_name: str) -> None:
        self._server_name: str = server_name
        self._topic_name: str  = topic_name
        self._server           = None
        self._conv             = None

    @property
    def connected(self) -> bool:
        """True if a DDE conversation is open."""
        return self._conv is not None

    def connect(self) -> bool:
        """
        Create a DDE server and open a conversation with OMNIC.

        Returns
        -------
        bool
            True on success.
        """
        srv = None
        try:
            srv  = dde.CreateServer()
            srv.Create(self._server_name)
            conv = dde.CreateConversation(srv)
            conv.ConnectTo(self._server_name, self._topic_name)
            self._server = srv
            self._conv   = conv
            logging.info("Spectrometer DDE connected: %s / %s",
                         self._server_name, self._topic_name)
            return True
        except Exception as exc:
            logging.error("Spectrometer connect failed: %s", exc)
            # Destroy the server if it was created but ConnectTo failed;
            # leaving it alive prevents a clean retry.
            if srv is not None:
                try:
                    srv.Destroy()
                except Exception:
                    pass
            self._server = None
            self._conv   = None
            return False

    def disconnect(self) -> None:
        """Drop DDE references without closing OMNIC."""
        self._conv = None
        if self._server is not None:
            try:
                self._server.Destroy()
            except Exception:
                pass
            self._server = None

    def _exec(self, cmd: str) -> None:
        """Send a DDE execute command; raises RuntimeError if not connected."""
        if self._conv is None:
            raise RuntimeError("Spectrometer not connected")
        self._conv.Exec(cmd)

    def set_option(self, name: str, value: str) -> None:
        """Set an OMNIC Options parameter via DDE.

        Uses the DDE ``Set`` command syntax::

            [Set Options <name> <value>]

        Parameters
        ----------
        name : str
            Option parameter name (e.g. ``'CollectPrompt'``).
        value : str
            New value (e.g. ``'True'`` or ``'False'``).
        """
        self._exec(f'[Set Options {name} {value}]')

    def load_experiment(self, exp_path: str) -> None:
        """
        Load an OMNIC experiment parameter file.

        Parameters
        ----------
        exp_path : str
            Full path to the ``.exp`` parameter file.
        """
        self._exec(f'[LoadParameters "{exp_path}"]')

    def bench_align(self) -> None:
        """Trigger the OMNIC bench alignment routine."""
        self._exec('[Invoke StartBenchAlign]')

    def start_collect(self, name: str) -> None:
        """
        Send the CollectSample DDE execute command.

        Uses ``Auto Polling`` for all modes: no OMNIC prompts, no collection
        window; spectrum is placed directly in the active spectral window.
        Shutters are left in manual/always-open mode in the ``.exp`` file.
        T/R purge equilibration is handled by the GUI before this call.

        Must be called from the main thread (Win32 DDE requires a message pump).
        Returns immediately; use :meth:`poll_collect_status` to track progress.

        Parameters
        ----------
        name : str
            Spectrum label passed to OMNIC.
        """
        self._exec(f'[CollectSample "{name}" Auto Polling]')

    def poll_collect_status(self) -> tuple[int, int]:
        """
        Request the current collection progress from OMNIC.

        Must be called from the main thread (Win32 DDE requires a message pump).

        Returns
        -------
        tuple[int, int]
            ``(n_scans_completed, pct_complete)`` where *pct_complete* is 0–100.
        """
        if self._conv is None:
            raise RuntimeError("Spectrometer not connected")
        status = self._conv.Request('Collect Status')
        parts  = status.split(',')
        return int(parts[0]), int(parts[7])

    def export_csv(self, out_path: str) -> None:
        """
        Export the currently displayed spectrum to a CSV file.

        Parameters
        ----------
        out_path : str
            Destination file path (OMNIC requires the ``.CSV`` extension).
        """
        self._exec(f'[Export "{out_path}"]')

    def display(self) -> None:
        """Display the most recently collected spectrum in OMNIC."""
        self._exec('[Display]')

    def hide_selected(self) -> None:
        """Hide the selected spectrum in the OMNIC spectral window."""
        self._exec('[HideSelectedSpectra]')

    def query_exp_params(self) -> dict:
        """
        Query current experiment parameters from OMNIC via DDE Request.

        Must be called from the main thread.  Failures on individual
        parameters are silently stored as empty strings.

        Returns
        -------
        dict
            Mapping of parameter key → string value.
        """
        if self._conv is None:
            return {}
        requests = [
            ('resolution',   'Collect Resolution'),
            ('num_scans',    'Collect NumScans'),
            ('apodization',  'Collect ApodizationFunction'),
            ('zero_fill',    'Collect ZeroFill'),
            ('high_cutoff',  'Bench HighCutoff'),
            ('low_cutoff',   'Bench LowCutoff'),
            ('gain',         'Bench Gain'),
            ('beamsplitter', 'Bench BeamSplitter'),
            ('velocity',     'Bench Velocity'),
        ]
        params: dict = {}
        for key, dde_param in requests:
            try:
                params[key] = self._conv.Request(dde_param).strip()
            except Exception:
                params[key] = ''
        return params


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scan_exp_files(mode: str) -> list[str]:
    """
    Return filenames of OMNIC ``.exp`` files matching *mode* from
    ``OMNIC_PARAM_DIR``.

    Keyword matching is case-insensitive.  Falls back to the mode-specific
    default filename if the directory is absent or yields no matches.

    Parameters
    ----------
    mode : str
        One of ``'Emission'``, ``'Transmission'``, ``'Reflectance'``.

    Returns
    -------
    list of str
        Sorted list of ``.exp`` filenames (not full paths).
    """
    keywords = _MODE_EXP_KEYWORDS.get(mode, ['emission'])
    default  = _MODE_EXP_DEFAULTS.get(mode, DEFAULT_EXP_FILENAME)
    if OMNIC_PARAM_DIR.exists():
        matches = sorted(
            p.name for p in OMNIC_PARAM_DIR.glob('*.exp')
            if any(kw in p.name.lower() for kw in keywords)
        )
        if matches:
            return matches
    return [default]


# ---------------------------------------------------------------------------
# Purge equilibration countdown dialog (T/R modes)
# ---------------------------------------------------------------------------

class _BBTempsDialog(tk.Toplevel):
    """Modal popup for manually entering BB temperatures when resistance data
    is unavailable (multimeter was not connected during BB collection).

    Each field accepts either a temperature in °C (single float) or a
    resistance pair ``"ch1 ch2"`` or ``"ch1, ch2"`` (converted via r2t_nau).

    Attributes
    ----------
    result : tuple[float, float] | None
        ``(bb_warm_K, bb_hot_K)`` on confirmation, ``None`` if cancelled.
    """

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title('Blackbody Temperatures')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: 'tuple[float, float] | None' = None
        self._build()
        self.wait_window(self)

    @staticmethod
    def _parse(raw: str, field: str) -> float:
        """Parse ``raw`` as °C (single value) or resistance pair → Kelvin."""
        parts = raw.replace(',', ' ').split()
        if len(parts) == 1:
            return c2k(float(parts[0]))
        elif len(parts) == 2:
            return r2t_nau(float(parts[0]), float(parts[1]))
        raise ValueError(
            f"{field}: expected a temperature in °C or a resistance pair (ch1 ch2)")

    def _build(self) -> None:
        frm = ttk.Frame(self, padding=16)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text='BB temperature data is missing from the measurement log.\n'
                 'The multimeter was likely not connected during BB collection.\n\n'
                 'Enter the blackbody temperatures manually.',
            foreground='#c05000',
            justify=tk.LEFT,
            wraplength=360,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        ttk.Label(
            frm,
            text='Enter a temperature in °C or a resistance pair as "ch1 ch2".',
            foreground='#666666',
            justify=tk.LEFT,
            wraplength=360,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        ttk.Label(frm, text='BB Warm:').grid(
            row=2, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._warm_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._warm_var, width=20).grid(
            row=2, column=1, sticky=tk.W)

        ttk.Label(frm, text='BB Hot:').grid(
            row=3, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        self._hot_var = tk.StringVar()
        ttk.Entry(frm, textvariable=self._hot_var, width=20).grid(
            row=3, column=1, sticky=tk.W)

        bf = ttk.Frame(frm)
        bf.grid(row=4, column=0, columnspan=2, pady=(14, 0))
        ttk.Button(bf, text='OK',     command=self._ok).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(side=tk.LEFT, padx=6)

        self.bind('<Return>', lambda _e: self._ok())
        self.bind('<Escape>', lambda _e: self.destroy())

    def _ok(self) -> None:
        try:
            warm_k = self._parse(self._warm_var.get().strip(), 'BB Warm')
            hot_k  = self._parse(self._hot_var.get().strip(),  'BB Hot')
            self.result = (warm_k, hot_k)
            self.destroy()
        except ValueError as exc:
            messagebox.showerror('Invalid input', str(exc), parent=self)


# ---------------------------------------------------------------------------

class _BBCollectDialog(tk.Toplevel):
    """Modal confirmation dialog for Collect BB.

    Displays the current estimated BB temperature, auto-selects BB Warm or
    BB Hot based on the reading, and lets the user override the choice or
    select the generic auto-numbered option.

    Parameters
    ----------
    master : tk.Misc
        Parent widget.
    bb_type_var : tk.StringVar
        Shared variable holding ``'bbwarm'``, ``'bbhot'``, or
        ``'bbgeneric'``.  The dialog reads *and writes* this var so the
        last manual choice persists across calls; the auto-selection only
        overrides it when a clear temperature signal is available.
    bb_temp_var : tk.StringVar
        Live-updated StringVar with the formatted BB temperature (e.g.
        ``'342.15'`` or ``'—'``).  Bound to the display label so any
        live-poll update during the dialog is reflected.

    Attributes
    ----------
    result : str | None
        ``'bbwarm'`` / ``'bbhot'`` / ``'bbgeneric'`` on confirmation,
        ``None`` if cancelled.
    """

    def __init__(
        self,
        master: tk.Misc,
        bb_type_var: tk.StringVar,
        bb_temp_var: tk.StringVar,
    ) -> None:
        super().__init__(master)
        self.title('Blackbody Collection')
        self.resizable(False, False)
        self.grab_set()
        self.transient(master)
        self.result: 'str | None' = None

        self._bb_type_var = bb_type_var

        # ── Auto-select based on current temperature ─────────────────────
        try:
            temp_c = float(bb_temp_var.get())
            if temp_c >= _BB_TEMP_HOT_MIN_C:
                bb_type_var.set('bbhot')
            elif temp_c >= _BB_TEMP_WARM_MIN_C:
                bb_type_var.set('bbwarm')
            # below threshold: leave selection unchanged (BB not at temperature)
        except (ValueError, TypeError):
            pass   # '—' or 'err' — leave unchanged

        frm = ttk.Frame(self, padding=20)
        frm.pack(fill=tk.BOTH, expand=True)

        # ── Temperature display ──────────────────────────────────────────
        ttk.Label(frm, text='Current BB temperature estimate:').grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))

        temp_frm = ttk.Frame(frm)
        temp_frm.grid(row=1, column=0, columnspan=2, pady=(0, 16))
        ttk.Label(
            temp_frm,
            textvariable=bb_temp_var,
            font=('TkDefaultFont', 22, 'bold'),
            anchor=tk.E,
            width=8,
        ).pack(side=tk.LEFT)
        ttk.Label(
            temp_frm,
            text=' °C',
            font=('TkDefaultFont', 16),
            anchor=tk.W,
        ).pack(side=tk.LEFT)

        # ── BB type selector ─────────────────────────────────────────────
        ttk.Label(frm, text='Blackbody type:').grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))
        ttk.Radiobutton(
            frm, text='BB Warm  (bbwarm)',
            variable=self._bb_type_var, value='bbwarm',
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=(8, 0))
        ttk.Radiobutton(
            frm, text='BB Hot  (bbhot)',
            variable=self._bb_type_var, value='bbhot',
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=(8, 0))
        ttk.Radiobutton(
            frm, text='BB (generic)',
            variable=self._bb_type_var, value='bbgeneric',
        ).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=(8, 0))
        ttk.Label(
            frm,
            text='Auto-numbered: bb001, bb002, …  (not used by emcal)',
            foreground='#888888',
            font=('TkDefaultFont', 9),
        ).grid(row=6, column=0, columnspan=2, sticky=tk.W, padx=(28, 0),
               pady=(0, 4))

        # ── Action buttons ───────────────────────────────────────────────
        bf = ttk.Frame(frm)
        bf.grid(row=7, column=0, columnspan=2, pady=(16, 0))
        ttk.Button(bf, text='Collect', command=self._on_collect).pack(
            side=tk.LEFT, padx=8)
        ttk.Button(bf, text='Cancel', command=self.destroy).pack(
            side=tk.LEFT, padx=8)

        self.bind('<Return>', lambda _e: self._on_collect())
        self.bind('<Escape>', lambda _e: self.destroy())

        self.wait_window(self)

    def _on_collect(self) -> None:
        self.result = self._bb_type_var.get()
        self.destroy()


# ---------------------------------------------------------------------------

class _PurgeCountdownDialog(tk.Toplevel):
    """Dismissible countdown shown before every T/R collection.

    Gives the sample compartment time to re-purge after being opened to insert
    a sample or accessory.  Auto-proceeds when the countdown reaches zero; the
    user can click "Proceed Now" to skip the remaining delay at any time.

    Parameters
    ----------
    parent : tk.Misc
        The main application window (used for centering and ``transient``).
    delay_s : int
        Initial countdown value in seconds (``PURGE_DELAY_S``).
    on_proceed : callable
        Called on the main thread exactly once — at countdown expiry or when
        the user dismisses the dialog.
    """

    def __init__(
        self,
        parent: tk.Misc,
        delay_s: int,
        on_proceed: 'Callable[[], None]',
    ) -> None:
        super().__init__(parent)
        self.title('Purge Equilibration')
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self._remaining  = delay_s
        self._on_proceed = on_proceed
        self._done       = False        # guard: fire callback exactly once

        # ── Layout ────────────────────────────────────────────────────────────
        outer = ttk.Frame(self, padding=20)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            outer,
            text=(
                'Sample compartment opened.\n'
                'Waiting for purge to equilibrate\n'
                'before starting the collection.'
            ),
            justify=tk.CENTER,
            anchor=tk.CENTER,
        ).pack(pady=(0, 12))

        self._timer_var = tk.StringVar(value=self._fmt(delay_s))
        ttk.Label(
            outer,
            textvariable=self._timer_var,
            font=('TkDefaultFont', 28, 'bold'),
            anchor=tk.CENTER,
        ).pack(pady=(0, 14))

        ttk.Button(outer, text='Proceed Now', command=self._proceed).pack()

        # Window-close (×) also proceeds rather than aborting the collection.
        self.protocol('WM_DELETE_WINDOW', self._proceed)

        # Centre over parent.
        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width()  - self.winfo_width())  // 2
        py = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f'+{px}+{py}')

        self._tick()

    @staticmethod
    def _fmt(seconds: int) -> str:
        return f'{seconds // 60:01d}:{seconds % 60:02d}'

    def _tick(self) -> None:
        if self._done:
            return
        self._timer_var.set(self._fmt(self._remaining))
        if self._remaining <= 0:
            self._proceed()
            return
        self._remaining -= 1
        self.after(1000, self._tick)

    def _proceed(self) -> None:
        if self._done:
            return
        self._done = True
        self.grab_release()
        self.destroy()
        self._on_proceed()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GUI log handler
# ---------------------------------------------------------------------------

class _GuiLogHandler(logging.Handler):
    """Routes log records to a ``tk.Text`` widget with colour-coded levels.

    Thread-safe: ``emit()`` may be called from background threads; all widget
    mutations are scheduled on the Tk main thread via ``widget.after()``.
    """

    _LEVEL_TAG: dict[int, str] = {
        logging.DEBUG:    'debug',
        logging.INFO:     'info',
        logging.WARNING:  'warning',
        logging.ERROR:    'error',
        logging.CRITICAL: 'error',
    }

    def __init__(self, text_widget: tk.Text) -> None:
        super().__init__()
        self._text = text_widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tag = self._LEVEL_TAG.get(record.levelno, 'info')
            self._text.after(0, self._append, msg, tag)
        except Exception:
            self.handleError(record)

    def _append(self, msg: str, tag: str) -> None:
        """Main-thread callback: insert one line and scroll to the bottom."""
        try:
            self._text.config(state='normal')
            self._text.insert(tk.END, msg + '\n', tag)
            self._text.see(tk.END)
            self._text.config(state='disabled')
        except tk.TclError:
            pass   # widget already destroyed — ignore


# Main window
# ---------------------------------------------------------------------------

class AutomateFTIR(tk.Tk):
    """
    Top-level application window for multi-mode FTIR data collection.

    Attributes
    ----------
    _ax : matplotlib.axes.Axes
        Primary spectral display axes (wavenumber, reversed).
    _secax : matplotlib.axes.Axes or None
        Secondary wavelength axis attached to the top of *_ax*.
    _save_fdir : Path or None
        Destination folder for saved spectra and log CSV.  None until chosen.
    _mode_var : tk.StringVar
        Active measurement mode: ``'Emission'``, ``'Transmission'``, or
        ``'Reflectance'``.
    _progress_var : tk.DoubleVar
        Drives the progress bar (range 0–100).
    _status_var : tk.StringVar
        Short status message shown next to the progress bar.
    _plot_mode_var : tk.StringVar
        Spectral display mode: ``'stacked'`` or ``'single'``.
    _scale_mode_var : tk.StringVar
        Y-axis scaling: ``'common'`` (shared limits) or ``'normalized'``.
    _mm_mode_var : tk.StringVar
        Multimeter display mode: ``'live'`` or ``'sample'``.
    _channel_vars : dict[int, tk.StringVar]
        Live-updated display values for each multimeter channel.
    _mm : MultimeterController
        Keithley 2700 connection manager.
    _spec : SpectrometerController
        OMNIC DDE connection manager.
    _spectra_data : dict[str, dict]
        In-memory store of collected spectra keyed by sample name.
    _bkg_spectrum : dict or None
        Background spectrum for Transmission / Reflectance processing.
    _spectrum_count : int
        Running row counter for the measurement log CSV.
    _collection_active : bool
        Prevents concurrent collection runs.
    _active_collection_mode : str
        Mode that was active when the current collection was started.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title(f'AutomateFTIR v{__version__}')
        self.state('zoomed')   # maximised on Windows
        self.minsize(1000, 640)

        # ── Save directory ────────────────────────────────────────────────
        self._save_fdir: Path | None = None
        self._save_fdir_var          = tk.StringVar(
            value='(select folder to enable collection)')

        # ── Mode ──────────────────────────────────────────────────────────
        self._mode_var = tk.StringVar(value='Emission')

        # ── Background spectrum (Transmission / Reflectance modes) ────────
        self._bkg_spectrum: dict | None = None

        # ── OMNIC experiment parameter file ───────────────────────────────
        self._exp_files = _scan_exp_files('Emission')
        _default = (DEFAULT_EXP_FILENAME
                    if DEFAULT_EXP_FILENAME in self._exp_files
                    else self._exp_files[0])
        self._exp_file_var = tk.StringVar(value=_default)

        # ── Control state ─────────────────────────────────────────────────
        self._progress_var           = tk.DoubleVar(value=0.0)
        self._status_var             = tk.StringVar(value='Ready')
        self._collection_active          = False
        self._collection_overwrite       = False
        self._active_collection_mode     = 'Emission'
        self._active_collection_is_blank = False
        self._active_collection_is_bkg   = False
        self._coll_start_time: 'datetime | None' = None   # set when DDE command succeeds
        self._spectrum_count             = 0
        self._last_mode                  = 'Emission'   # for mode-change revert on cancel

        # ── Plot controls ─────────────────────────────────────────────────
        self._plot_mode_var       = tk.StringVar(value='stacked')
        self._scale_mode_var      = tk.StringVar(value='common')
        self._display_var         = tk.StringVar(value='sbm')
        self._color_scheme_var    = tk.StringVar(value=_COLOR_SCHEME_DEFAULT)
        self._plot_colors: list   = list(_COLOR_SCHEMES[_COLOR_SCHEME_DEFAULT])

        # ── Experiment parameter cache ────────────────────────────────────
        self._exp_params: dict = {}

        # ── emcal state ───────────────────────────────────────────────────
        self._emcal_opts: dict = dict(
            lab='nau', method='nem', max_emiss=1.0, bb_emiss=0.995,
            n_bb=2, temp_halfwidth=50.0, violation_weight=5.0, violation_tol=0.0,
            escalation_factor=4.0, max_escalations=4,
            noise_free=True, apply_dehyd=False,
            wn_range=(500.0, 1700.0),
        )
        self._purge_delay_var   = tk.IntVar(value=PURGE_DELAY_S)
        self._purge_enabled_var = tk.BooleanVar(value=True)
        self._bb_type_var      = tk.StringVar(value='bbwarm')   # 'bbwarm' | 'bbhot'
        self._emcal_result:    dict | None = None
        self._tracal_result: dict | None = None
        self._refcal_result:  dict | None = None

        # ── Multimeter display ────────────────────────────────────────────
        self._mm_mode_var    = tk.StringVar(value='live')
        self._live_readings: dict[int, float] = {}   # last live-poll snapshot
        self._channel_vars: dict[int, tk.StringVar] = {
            ch: tk.StringVar(value='—') for ch in _PANEL_ORDER
        }
        self._bb_temp_var         = tk.StringVar(value='—')
        self._mm_last_updated_var = tk.StringVar(value='—')

        # ── Measurement-mode MM sampling ──────────────────────────────────
        self._mm_measurement_samples:    dict[int, list[float]] = {}
        self._mm_measurement_timestamps: list[datetime]         = []
        self._stop_mm_measurement  = threading.Event()
        self._mm_sample_thread: threading.Thread | None = None

        # ── Hardware controllers ──────────────────────────────────────────
        self._mm   = MultimeterController(MULTIMETER_ADDRESS)
        self._spec = SpectrometerController(OMNIC_SERVER_NAME, OMNIC_TOPIC_NAME)

        # ── Spectra in-memory store ───────────────────────────────────────
        self._spectra_data: dict[str, dict] = {}

        # ── Live poll thread ──────────────────────────────────────────────
        self._poll_thread: threading.Thread | None = None
        self._stop_poll                            = threading.Event()

        # ── OMNIC auto-connect polling state ─────────────────────────────
        self._spec_polling:    bool  = False
        self._spec_poll_start: float = 0.0

        self.protocol('WM_DELETE_WINDOW', self._on_close)
        self._build_ui()
        self._refresh_buttons()

        # ── Logging setup ─────────────────────────────────────────────────
        logging.getLogger().setLevel(logging.DEBUG)

        # GUI panel — INFO and above, human-readable timestamp only.
        _gui_fmt = logging.Formatter(
            '%(asctime)s  [%(levelname)-7s]  %(message)s',
            datefmt='%H:%M:%S',
        )
        self._log_handler = _GuiLogHandler(self._log_text)
        self._log_handler.setFormatter(_gui_fmt)
        self._log_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._log_handler)

        # Rotating file log — DEBUG and above, full timestamps and module info.
        _LOG_PATH = Path(__file__).resolve().parent / 'AutomateFTIR.log'
        _file_fmt = logging.Formatter(
            '%(asctime)s  [%(levelname)-8s]  %(module)s.%(funcName)s  %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        self._file_log_handler = logging.handlers.RotatingFileHandler(
            _LOG_PATH,
            maxBytes=2 * 1024 * 1024,   # 2 MB per file
            backupCount=3,
            encoding='utf-8',
        )
        self._file_log_handler.setFormatter(_file_fmt)
        self._file_log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(self._file_log_handler)

        # Write a session-start banner directly to the file (bypasses formatter
        # so it appears as a clean separator regardless of log level filtering).
        _session_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._file_log_handler.stream.write(
            f'\n{"─" * 72}\n'
            f'  Session started: {_session_start}  |  AutomateFTIR {__version__}\n'
            f'{"─" * 72}\n'
        )
        self._file_log_handler.stream.flush()

        logging.info("AutomateFTIR %s started — log: %s", __version__, _LOG_PATH)

        # Attempt auto-connect after the window is visible.
        self.after(600, self._auto_connect)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_top_frame()
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=(2, 0))
        self._build_main_frame()

    # ── Top frame ───────────────────────────────────────────────────────────

    def _build_top_frame(self) -> None:
        top = ttk.Frame(self, padding=(8, 6, 8, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        # ── Row 0: mode selector | OMNIC params | bench align ──────────────
        mode_row = ttk.Frame(top)
        mode_row.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(mode_row, text='Mode:').pack(side=tk.LEFT, padx=(0, 6))
        for mode in ('Emission', 'Transmission', 'Reflectance'):
            ttk.Radiobutton(
                mode_row, text=mode, variable=self._mode_var,
                value=mode, command=self._on_mode_change,
            ).pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(mode_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        ttk.Label(mode_row, text='OMNIC Params:').pack(side=tk.LEFT, padx=(0, 4))
        self._cb_exp = ttk.Combobox(
            mode_row,
            textvariable=self._exp_file_var,
            values=self._exp_files,
            state='readonly',
            width=36,
        )
        self._cb_exp.pack(side=tk.LEFT, padx=(0, 4))
        self._cb_exp.bind('<<ComboboxSelected>>', self._on_exp_file_change)
        self._btn_load_params = ttk.Button(
            mode_row, text='Load', command=self._spec_auto_load_params,
            state='disabled')
        self._btn_load_params.pack(side=tk.LEFT, padx=(0, 4))
        self._btn_show_params = ttk.Button(
            mode_row, text='Show Params', command=self._on_show_params,
            state='disabled')
        self._btn_show_params.pack(side=tk.LEFT, padx=(0, 8))

        ttk.Separator(mode_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

        self._btn_bench_align = ttk.Button(
            mode_row, text='Bench Align', command=self._on_bench_align)
        self._btn_bench_align.pack(side=tk.LEFT, padx=(4, 0))

        # Purge delay — shown only in T/R modes, hidden in Emission.
        self._purge_frame = ttk.Frame(mode_row)
        ttk.Separator(self._purge_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=(10, 4), pady=2)
        ttk.Checkbutton(
            self._purge_frame,
            text='Purge delay:',
            variable=self._purge_enabled_var,
            command=self._on_purge_toggle,
        ).pack(side=tk.LEFT)
        self._purge_spinbox = ttk.Spinbox(
            self._purge_frame,
            textvariable=self._purge_delay_var,
            from_=0, to=300, increment=5, width=4,
        )
        self._purge_spinbox.pack(side=tk.LEFT, padx=(4, 2))
        ttk.Label(self._purge_frame, text='s').pack(side=tk.LEFT)
        # Not packed initially; _refresh_buttons shows/hides this frame.

        # ── Row 1: collection | name | processing | save folder ────────────
        btn_row = ttk.Frame(top)
        btn_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))

        # Collection buttons — order managed entirely by _refresh_buttons().
        self._btn_collect_bb = ttk.Button(
            btn_row, text='Collect BB', command=self._on_collect_bb)
        self._btn_collect_bkg = ttk.Button(
            btn_row, text='Collect Bkg', command=self._on_collect_bkg)
        self._btn_collect_blank = ttk.Button(
            btn_row, text='Collect Blank', command=self._on_collect_blank)
        self._btn_collect_sample = ttk.Button(
            btn_row, text='Collect Sample', command=self._on_collect_sample)

        # Name entry — packed by _refresh_buttons() immediately after Collect Sample.
        self._name_label = ttk.Label(btn_row, text='Name:')
        self._sample_name_entry = ttk.Entry(btn_row, width=22)

        # Separator between name entry and processing buttons.
        self._coll_proc_sep = ttk.Separator(btn_row, orient=tk.VERTICAL)

        # Processing buttons — one visible at a time, inside a sub-frame so
        # pack_forget/pack order stays predictable.
        self._proc_frame = ttk.Frame(btn_row)

        self._btn_emcal = ttk.Button(
            self._proc_frame, text='emcal()', command=self._on_emcal,
            state='disabled')

        self._btn_tracal = ttk.Button(
            self._proc_frame, text='tracal()', command=self._on_tracal,
            state='disabled')

        self._btn_refcal = ttk.Button(
            self._proc_frame, text='refcal()', command=self._on_refcal,
            state='disabled')

        # Right side — save folder (packed RIGHT, rightmost first)
        ttk.Button(
            btn_row, text='Browse…', command=self._on_browse_folder,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Entry(
            btn_row,
            textvariable=self._save_fdir_var,
            state='readonly',
        ).pack(side=tk.RIGHT, padx=(0, 4), expand=True, fill=tk.X)
        ttk.Label(btn_row, text='Save to:').pack(side=tk.RIGHT, padx=(4, 4))
        ttk.Separator(btn_row, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=2)

        # ── Row 2: status | progress | connect buttons + indicator lights ──
        info_row = ttk.Frame(top)
        info_row.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        ttk.Label(info_row, text='Status:').pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(
            info_row,
            textvariable=self._status_var,
            foreground='gray',
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 4), expand=True, fill=tk.X)

        # Right side — indicator lights with Connect buttons (rightmost first)
        self._light_multimeter = IndicatorLight(info_row)
        self._light_multimeter.pack(side=tk.RIGHT)
        ttk.Label(info_row, text='Multimeter:').pack(side=tk.RIGHT, padx=(4, 2))
        ttk.Button(
            info_row, text='Connect', width=8,
            command=self._connect_multimeter,
        ).pack(side=tk.RIGHT, padx=(0, 4))

        ttk.Separator(info_row, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=2)

        self._light_spectrometer = IndicatorLight(info_row)
        self._light_spectrometer.pack(side=tk.RIGHT)
        ttk.Label(info_row, text='Spectrometer:').pack(side=tk.RIGHT, padx=(4, 2))
        ttk.Button(
            info_row, text='Connect', width=8,
            command=self._connect_spectrometer,
        ).pack(side=tk.RIGHT, padx=(0, 4))

        ttk.Separator(info_row, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=2)

        # Progress bar and scan counter — just left of the connect buttons
        self._scan_label = ttk.Label(info_row, text='', width=26, anchor=tk.W)
        self._scan_label.pack(side=tk.RIGHT)

        self._progress_bar = ttk.Progressbar(
            info_row,
            variable=self._progress_var,
            mode='determinate',
            maximum=100,
            length=750,
        )
        self._progress_bar.pack(side=tk.RIGHT, padx=(8, 4))

    # ── Main area ────────────────────────────────────────────────────────────

    def _build_main_frame(self) -> None:
        main = ttk.Frame(self, padding=(6, 4))
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        self._build_plot_panel(paned)
        self._build_right_panel(paned)

    def _build_plot_panel(self, parent: ttk.PanedWindow) -> None:
        plot_frame = ttk.Frame(parent)
        parent.add(plot_frame, weight=4)

        # ── Nav bar: display mode + scale mode ────────────────────────────
        nav = ttk.Frame(plot_frame)
        nav.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 2))

        ttk.Radiobutton(
            nav, text='Stacked', variable=self._plot_mode_var,
            value='stacked', command=self._refresh_plot,
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Radiobutton(
            nav, text='Single', variable=self._plot_mode_var,
            value='single', command=self._refresh_plot,
        ).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Separator(nav, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        ttk.Radiobutton(
            nav, text='Common scale', variable=self._scale_mode_var,
            value='common', command=self._refresh_plot,
        ).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Radiobutton(
            nav, text='Normalized', variable=self._scale_mode_var,
            value='normalized', command=self._refresh_plot,
        ).pack(side=tk.LEFT)

        ttk.Separator(nav, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=8, pady=2)

        self._sbm_rb = ttk.Radiobutton(
            nav, text='Single Beam', variable=self._display_var,
            value='sbm', command=self._refresh_plot,
        )
        self._sbm_rb.pack(side=tk.LEFT, padx=(0, 2))

        # ── Emission display mode buttons (Radiance, Emissivity) ───────────
        self._radiance_rb = ttk.Radiobutton(
            nav, text='Radiance', variable=self._display_var,
            value='radiance', command=self._refresh_plot, state='disabled',
        )
        self._emissivity_rb = ttk.Radiobutton(
            nav, text='Emissivity', variable=self._display_var,
            value='emissivity', command=self._refresh_plot, state='disabled',
        )
        # Initially packed (Emission is the default mode).
        self._radiance_rb.pack(side=tk.LEFT, padx=(0, 2))
        self._emissivity_rb.pack(side=tk.LEFT, padx=(0, 2))

        # ── T/R display mode buttons (hidden until mode switches) ──────────
        # Transmission: Transmittance | Absorbance | Optical Depth
        self._transmittance_rb = ttk.Radiobutton(
            nav, text='Transmittance', variable=self._display_var,
            value='transmittance', command=self._refresh_plot, state='disabled',
        )
        self._absorbance_rb = ttk.Radiobutton(
            nav, text='Absorbance', variable=self._display_var,
            value='absorbance', command=self._refresh_plot, state='disabled',
        )
        self._od_rb = ttk.Radiobutton(
            nav, text='Opt. Depth', variable=self._display_var,
            value='od', command=self._refresh_plot, state='disabled',
        )
        # Reflectance
        self._reflectance_rb = ttk.Radiobutton(
            nav, text='Reflectance', variable=self._display_var,
            value='reflectance', command=self._refresh_plot, state='disabled',
        )
        # Kirchhoff complement (1 − E/T/R) — shared across all modes;
        # label text is updated by _refresh_buttons before packing.
        self._kirchhoff_rb = ttk.Radiobutton(
            nav, text='1−E (Kirchhoff)', variable=self._display_var,
            value='kirchhoff', command=self._refresh_plot, state='disabled',
        )
        # All start hidden; _refresh_buttons packs the right ones.

        # Color scheme selector (right side of nav bar)
        cb_scheme = ttk.Combobox(
            nav,
            textvariable=self._color_scheme_var,
            values=list(_COLOR_SCHEMES.keys()),
            state='readonly',
            width=10,
        )
        cb_scheme.pack(side=tk.RIGHT, padx=(0, 4))
        cb_scheme.bind('<<ComboboxSelected>>', self._on_color_scheme_change)
        ttk.Label(nav, text='Color Scheme:').pack(side=tk.RIGHT, padx=(0, 2))

        # ── Matplotlib canvas + toolbar (expanding inner frame) ───────────
        canvas_frame = ttk.Frame(plot_frame)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(10, 5), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(top=0.87, bottom=0.10, left=0.08, right=0.97)

        self._ax.set_xlabel('Wavenumber (cm⁻\xb9)')
        self._ax.set_ylabel('Single Beam')
        self._ax.set_xlim(2000, 200)

        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, canvas_frame)

        self._secax = _add_top_axis(self._ax)
        self._canvas.draw()

        # ── Log panel ─────────────────────────────────────────────────────
        log_lf = ttk.LabelFrame(plot_frame, text='Log', padding=(4, 2))
        log_lf.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(4, 4))

        log_scroll = ttk.Scrollbar(log_lf, orient=tk.VERTICAL)
        self._log_text = tk.Text(
            log_lf,
            height=8,
            state='disabled',
            font=('TkFixedFont', 9),
            yscrollcommand=log_scroll.set,
            wrap=tk.WORD,
        )
        log_scroll.config(command=self._log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Colour tags — applied per log level.
        self._log_text.tag_config('debug',   foreground='#999999')
        self._log_text.tag_config('info',    foreground='#555555')
        self._log_text.tag_config('warning', foreground='#bb7700')
        self._log_text.tag_config('error',   foreground='#cc2200')

    # ── Right panel ──────────────────────────────────────────────────────────

    def _build_right_panel(self, parent: ttk.PanedWindow) -> None:
        right = ttk.Frame(parent, width=380)
        right.pack_propagate(False)
        parent.add(right, weight=0)

        # ── Collected spectra listbox ──────────────────────────────────────
        spec_lf = ttk.LabelFrame(right, text='Collected Spectra', padding=4)
        spec_lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=(2, 4))

        lb_frame = ttk.Frame(spec_lf)
        lb_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        self._spectra_lb = tk.Listbox(
            lb_frame,
            yscrollcommand=sb.set,
            selectmode=tk.SINGLE,
            exportselection=False,
            font=('TkFixedFont', 11),
            activestyle='dotbox',
        )
        sb.config(command=self._spectra_lb.yview)
        self._spectra_lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._spectra_lb.bind('<<ListboxSelect>>', self._on_spectra_select)

        btn_spec_row = ttk.Frame(spec_lf)
        btn_spec_row.pack(side=tk.TOP, fill=tk.X, pady=(4, 0))
        ttk.Button(
            btn_spec_row, text='Delete Single',
            command=self._on_delete_spectrum,
        ).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(
            btn_spec_row, text='Delete All',
            command=self._on_delete_all_spectra,
        ).pack(side=tk.LEFT)

        # ── Multimeter section ─────────────────────────────────────────────
        self._build_multimeter_panel(right)

    def _build_multimeter_panel(self, parent: tk.Widget) -> None:
        mm_frame = ttk.LabelFrame(parent, text='Multimeter', padding=(10, 6))
        mm_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(0, 2))
        # Column 0: channel labels (fixed); column 1: values (expands to fill
        # available width); column 2: units (fixed).
        mm_frame.columnconfigure(0, weight=0)
        mm_frame.columnconfigure(1, weight=1)
        mm_frame.columnconfigure(2, weight=0)

        # ── Mode controls ──────────────────────────────────────────────────
        ctrl_row = ttk.Frame(mm_frame)
        ctrl_row.grid(row=0, column=0, columnspan=3, sticky=tk.EW, pady=(0, 2))

        ttk.Radiobutton(
            ctrl_row, text='Live', variable=self._mm_mode_var, value='live',
            command=self._on_mm_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(
            ctrl_row, text='Sample', variable=self._mm_mode_var,
            value='sample', command=self._on_mm_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 10))
        self._btn_refresh = ttk.Button(
            ctrl_row, text='Refresh', command=self._on_refresh, width=9)
        self._btn_refresh.pack(side=tk.RIGHT, padx=(4, 0))

        ttk.Separator(mm_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=3, sticky=tk.EW, pady=(4, 6))

        grid_row = 2
        for ch in _PANEL_ORDER:
            ttk.Label(mm_frame, text=f'{_CHANNEL_LABELS[ch]}:', anchor=tk.W).grid(
                row=grid_row, column=0, sticky=tk.W, padx=(0, 4), pady=2)
            ttk.Label(
                mm_frame,
                textvariable=self._channel_vars[ch],
                anchor=tk.E,
                font=('TkFixedFont', 11),
            ).grid(row=grid_row, column=1, sticky=tk.EW, padx=(0, 4), pady=2)
            ttk.Label(
                mm_frame,
                text=_CHANNEL_UNITS[ch],
                anchor=tk.W,
                foreground='gray',
            ).grid(row=grid_row, column=2, sticky=tk.W, pady=2)
            grid_row += 1

            if ch == _PANEL_SEPARATOR_AFTER:
                ttk.Label(
                    mm_frame, text='BB temp (est.):', anchor=tk.W, foreground='gray',
                ).grid(row=grid_row, column=0, sticky=tk.W, padx=(0, 4), pady=(2, 2))
                ttk.Label(
                    mm_frame,
                    textvariable=self._bb_temp_var,
                    anchor=tk.E,
                    font=('TkFixedFont', 11),
                    foreground='gray',
                ).grid(row=grid_row, column=1, sticky=tk.EW, padx=(0, 4), pady=(2, 2))
                ttk.Label(
                    mm_frame, text='°C', anchor=tk.W, foreground='gray',
                ).grid(row=grid_row, column=2, sticky=tk.W, pady=(2, 2))
                grid_row += 1

                ttk.Separator(mm_frame, orient=tk.HORIZONTAL).grid(
                    row=grid_row, column=0, columnspan=3,
                    sticky=tk.EW, pady=(4, 4))
                grid_row += 1

        # ── Last-updated timestamp ─────────────────────────────────────────
        ttk.Separator(mm_frame, orient=tk.HORIZONTAL).grid(
            row=grid_row, column=0, columnspan=3, sticky=tk.EW, pady=(6, 2))
        grid_row += 1
        ttk.Label(
            mm_frame, text='Last updated:', anchor=tk.W,
            foreground='gray', font=('TkDefaultFont', 9),
        ).grid(row=grid_row, column=0, sticky=tk.W, padx=(0, 4), pady=(2, 2))
        ttk.Label(
            mm_frame, textvariable=self._mm_last_updated_var,
            anchor=tk.E, foreground='gray',
            font=('TkFixedFont', 9),
        ).grid(row=grid_row, column=1, columnspan=2, sticky=tk.EW, pady=(2, 2))

    # -----------------------------------------------------------------------
    # Connection management
    # -----------------------------------------------------------------------

    def _auto_connect(self) -> None:
        """Connect to the multimeter; connect to OMNIC if already open, else launch it."""
        self._connect_multimeter()
        # Try DDE first — if OMNIC is already running this succeeds immediately.
        logging.info("Attempting spectrometer connection via DDE…")
        if self._spec.connect():
            self._light_spectrometer.set_state('ok')
            logging.info("Spectrometer connected via DDE")
            self._spec_auto_load_params()
            return
        # OMNIC not yet open — launch it and start polling.
        try:
            subprocess.Popen([str(OMNIC_EXE)])
            logging.info("OMNIC launched: %s", OMNIC_EXE)
        except Exception as exc:
            logging.error("Failed to launch OMNIC: %s", exc)
            self._status_var.set(
                'Could not launch OMNIC — open it manually, then click Connect')
            return
        self._spec_polling    = True
        self._spec_poll_start = time.monotonic()
        self._status_var.set(
            f'OMNIC launching… waiting up to {OMNIC_AUTOCONNECT_TIMEOUT_S}s')
        self.after(OMNIC_AUTOCONNECT_POLL_MS, self._spec_poll_connect)

    def _spec_poll_connect(self) -> None:
        """Main thread: retry DDE connect every poll interval until ready or timed out."""
        if not self._spec_polling:
            return  # cancelled by a manual Connect click
        if self._spec.connect():
            self._spec_polling = False
            self._light_spectrometer.set_state('ok')
            logging.info("Spectrometer connected via DDE")
            self._spec_auto_load_params()
            return
        elapsed   = time.monotonic() - self._spec_poll_start
        remaining = int(OMNIC_AUTOCONNECT_TIMEOUT_S - elapsed)
        if remaining <= 0:
            self._spec_polling = False
            self._light_spectrometer.set_state('error')
            self._status_var.set(
                f'OMNIC did not respond in {OMNIC_AUTOCONNECT_TIMEOUT_S}s — '
                'check OMNIC then click Connect')
            logging.warning("OMNIC auto-connect timed out after %.0fs", elapsed)
            return
        self._status_var.set(f'Waiting for OMNIC… ({remaining}s remaining)')
        self.after(OMNIC_AUTOCONNECT_POLL_MS, self._spec_poll_connect)

    def _spec_auto_load_params(self) -> None:
        """Load the currently selected .exp file and query experiment parameters."""
        exp_file = self._exp_file_var.get()
        exp_path = str(OMNIC_PARAM_DIR / exp_file)
        try:
            self._spec.load_experiment(exp_path)
            self._status_var.set(f'Ready — parameters loaded: {exp_file}')
            logging.info("OMNIC parameters loaded: %s", exp_path)
        except Exception as exc:
            logging.error("Auto load params failed: %s", exc)
            self._status_var.set('Spectrometer connected — parameter load failed')
            return
        params = self._spec.query_exp_params()
        params['exp_file'] = exp_file
        self._exp_params = params
        self._btn_load_params.config(state='normal')
        self._btn_show_params.config(state='normal')

    def _connect_multimeter(self) -> None:
        """
        Try to connect to the Keithley; update the indicator light and,
        if in live mode, start the background polling thread.
        """
        self._status_var.set('Connecting to multimeter…')
        self.update_idletasks()
        if self._mm.connect():
            self._light_multimeter.set_state('ok')
            self._status_var.set('Multimeter connected')
            logging.info("Multimeter connected")
            # Immediate first read so the display is populated right away.
            threading.Thread(target=self._refresh_worker, daemon=True).start()
            if self._mm_mode_var.get() == 'live':
                self._start_live_poll()
        else:
            self._light_multimeter.set_state('error')
            self._status_var.set('Multimeter connection failed')
            logging.error("Multimeter connection failed")

    def _connect_spectrometer(self) -> None:
        """Try to connect to OMNIC via DDE; cancel any auto-connect polling first."""
        self._spec_polling = False  # stop the auto-connect loop if running
        self._status_var.set('Connecting to spectrometer…')
        self.update_idletasks()
        if self._spec.connect():
            self._light_spectrometer.set_state('ok')
            logging.info("Spectrometer connected via DDE")
            self._spec_auto_load_params()
        else:
            self._light_spectrometer.set_state('error')
            self._status_var.set('Spectrometer connection failed — is OMNIC open?')
            logging.error("Spectrometer connection failed — OMNIC not responding")

    # -----------------------------------------------------------------------
    # Live poll
    # -----------------------------------------------------------------------

    def _start_live_poll(self) -> None:
        """Launch the background multimeter polling thread if not already running."""
        if self._poll_thread is not None and self._poll_thread.is_alive():
            return
        self._stop_poll.clear()
        self._poll_thread = threading.Thread(
            target=self._live_poll_loop, daemon=True)
        self._poll_thread.start()

    def _stop_live_poll(self) -> None:
        """Signal the polling thread to exit at its next wake-up."""
        self._stop_poll.set()

    def _live_poll_loop(self) -> None:
        """
        Background thread: read all channels every ``MM_POLL_INTERVAL_S``
        seconds and post the results to the Tk main thread via ``after``.

        Exits cleanly when ``_stop_poll`` is set or the multimeter disconnects.
        """
        while not self._stop_poll.wait(MM_POLL_INTERVAL_S):
            if not self._mm.connected:
                break
            readings = self._mm.read_all_channels(list(_PANEL_ORDER))
            if not self._mm.connected:
                # A VisaIOError inside read_channel called disconnect(); stop polling.
                self.after(0, lambda: self._light_multimeter.set_state('error'))
                self.after(0, lambda: self._status_var.set(
                    'Multimeter connection lost — click Connect to reconnect'))
                break
            nan_chs = [ch for ch, v in readings.items() if np.isnan(v)]
            if nan_chs:
                logging.warning("Live poll: NaN on channels %s (transient read error)", nan_chs)
                self.after(0, lambda chs=nan_chs: self._status_var.set(
                    f'MM read warning — no data on channel(s) {chs}'))
            self.after(0, lambda r=readings: self._on_live_reading(r))

    def _on_live_reading(self, readings: dict[int, float]) -> None:
        """Main thread: store live snapshot, update display, append to live log."""
        self._live_readings = readings
        self.update_all_channels(readings)
        self._append_live_log(readings)
        if any(not np.isnan(v) for v in readings.values()):
            self._mm_last_updated_var.set(datetime.now().strftime('%H:%M:%S'))

    def _append_live_log(
        self,
        readings: dict[int, float],
        ts: datetime | None = None,
        sample_name: str = '',
        is_bb: str = '',
    ) -> None:
        """Append one timestamped row to the live multimeter log CSV.

        No-op in Transmission / Reflectance modes — multimeter data is not
        collected or logged outside Emission mode.
        """
        if self._save_fdir is None or self._mode_var.get() != 'Emission':
            return
        log_path = self._save_fdir / f'{self._save_fdir.name}-live-log.csv'
        try:
            self._save_fdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("Could not ensure save folder for live log: %s", exc)
            return
        is_new   = not log_path.exists()
        channels = sorted(readings.keys())
        headers  = ['dtime', 'sample_name', 'is_bb'] + [f'channel_{ch}' for ch in channels]
        row      = [(ts or datetime.now()).strftime('%H:%M:%S'), sample_name, is_bb] + [
            readings.get(ch, '') for ch in channels
        ]
        with open(log_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(headers)
            writer.writerow(row)

    def _on_mm_mode_change(self) -> None:
        """Start or stop live polling when the Live / Sample radio changes."""
        if self._mm_mode_var.get() == 'live':
            if self._mm.connected:
                self._start_live_poll()
            # Restore the last live snapshot, or blank if none yet.
            for ch in _PANEL_ORDER:
                v = self._live_readings.get(ch)
                self._channel_vars[ch].set('—' if v is None else f'{v:.2f}')
        else:
            self._stop_live_poll()
            self._display_selected_spectrum_channels()

    def _display_selected_spectrum_channels(self) -> None:
        """Show mm_stats for the selected listbox spectrum in Sample mode, or '—'."""
        sel = self._spectra_lb.curselection()
        if not sel:
            for ch in _PANEL_ORDER:
                self._channel_vars[ch].set('—')
            self._bb_temp_var.set('—')
            return
        name    = self._spectra_lb.get(sel[0])
        entry   = self._spectra_data.get(name, {})
        is_bb   = entry.get('is_bb', False)
        mm_stats = entry.get('mm_stats', {})
        for ch in _PANEL_ORDER:
            if ch in {101, 102} and not is_bb:
                self._channel_vars[ch].set('—')
                continue
            s = mm_stats.get(ch)
            if s and s.get('mean', '') != '':
                std = s.get('std', 0.0)
                val = f"{s['mean']:.2f} ± {std:.2f}" if std else f"{s['mean']:.2f}"
                self._channel_vars[ch].set(val)
            else:
                self._channel_vars[ch].set('—')
        # BB temp: show computed value for BB spectra, blank for samples
        if is_bb:
            s101 = mm_stats.get(101, {})
            s102 = mm_stats.get(102, {})
            r1 = s101.get('mean', '')
            r2 = s102.get('mean', '')
            if isinstance(r1, float) and isinstance(r2, float):
                try:
                    self._bb_temp_var.set(f'{r2t_nau(r1, r2) - 273.15:.2f}')
                except Exception:
                    self._bb_temp_var.set('err')
            else:
                self._bb_temp_var.set('—')
        else:
            self._bb_temp_var.set('—')

    # -----------------------------------------------------------------------
    # Plot helpers
    # -----------------------------------------------------------------------

    def _redraw_plot(self) -> None:
        """
        Rebuild the axes from the current display mode and stored data.

        Single Beam mode plots from ``_spectra_data``; Radiance mode plots
        from ``_emcal_result``.  Both respect the stacked/single toggle and
        the normalized scale mode.  BB spectra use fixed colors outside the
        sample color cycle.
        """
        self._secax = None
        self._ax.cla()
        self._ax.set_prop_cycle(color=self._plot_colors)
        self._ax.set_xlabel('Wavenumber (cm⁻\xb9)')

        normalized = self._scale_mode_var.get() == 'normalized'
        display    = self._display_var.get()

        if display in ('radiance', 'emissivity') and self._emcal_result is not None:
            if display == 'radiance':
                self._redraw_radiance(normalized)
            else:
                self._redraw_emissivity(normalized)
        elif display in ('transmittance', 'absorbance', 'od', 'reflectance'):
            self._redraw_ratio(normalized, display)
        elif display == 'kirchhoff':
            self._redraw_kirchhoff(normalized)
        else:
            self._redraw_sbm(normalized)

        self._secax = _add_top_axis(self._ax)
        self._canvas.draw_idle()

    def _redraw_sbm(self, normalized: bool) -> None:
        """Draw single-beam spectra from ``_spectra_data``.

        Line style / color rules
        ------------------------
        * Emission BB spectra   : fixed ``_BB_HOT_COLOR`` / ``_BB_WARM_COLOR``,
                                  ``--`` (hot) or ``:`` (warm).
        * T/R background spectra: ``--``, vibrant cycler (``_VIBRANT_BKG_COLORS``).
        * T/R blank spectra     : ``:``,  vibrant cycler.
        * Sample spectra        : solid,  standard color cycle (prop_cycle).
        """
        self._ax.set_ylabel('Single Beam (normalized)' if normalized else 'Single Beam')

        if not self._spectra_data:
            self._ax.set_xlim(2000, 200)
            return

        names_to_plot = self._names_for_current_mode(list(self._spectra_data))
        all_wn = np.concatenate([self._spectra_data[n]['wn'] for n in names_to_plot])
        self._ax.set_xlim(all_wn.max(), all_wn.min())

        vibrant  = _VIBRANT_BKG_COLORS
        vib_idx  = 0   # shared counter across bkg and blank entries

        for n in names_to_plot:
            entry      = self._spectra_data[n]
            data       = normalize(entry['data']) if normalized else entry['data']
            is_bb      = entry.get('is_bb',    False)
            is_bkg     = entry.get('is_bkg',   False)
            is_blank   = entry.get('is_blank', False)

            if is_bb:
                # Emission blackbody: fixed hot/warm colours
                name_lower = n.lower()
                color = _BB_HOT_COLOR if 'hot' in name_lower else _BB_WARM_COLOR
                ls    = '--'          if 'hot' in name_lower else ':'
                self._ax.plot(entry['wn'], data, label=n, color=color, ls=ls, lw=1.2)
            elif is_bkg:
                # T/R background reference: vibrant cycler, dashed
                color = vibrant[vib_idx % len(vibrant)]
                vib_idx += 1
                self._ax.plot(entry['wn'], data, label=n, color=color, ls='--', lw=1.2)
            elif is_blank:
                # T/R blank: vibrant cycler, dotted
                color = vibrant[vib_idx % len(vibrant)]
                vib_idx += 1
                self._ax.plot(entry['wn'], data, label=n, color=color, ls=':', lw=1.5)
            else:
                # Regular sample: solid line, drawn from the standard prop_cycle
                self._ax.plot(entry['wn'], data, label=n)

        self._ax.legend(fontsize=9, loc='upper left')

    def _redraw_radiance(self, normalized: bool) -> None:
        """Draw calibrated radiance spectra from ``_emcal_result``."""
        r   = self._emcal_result
        wn  = r.get('xaxis', np.array([]))
        rad = r.get('rad', {})

        if wn.size:
            self._ax.set_xlim(wn.max(), wn.min())
        self._ax.set_ylabel('Radiance (normalized)' if normalized
                            else 'Radiance (mW m⁻² sr⁻¹ cm)')

        all_sample_names = r.get('label', [])
        names_to_plot = self._names_for_current_mode(all_sample_names)

        for n in names_to_plot:
            if n not in rad:
                continue
            data = normalize(rad[n]) if normalized else rad[n]
            self._ax.plot(wn, data, label=n)

        # BB reference spectra from the result
        for key, color, ls, label in (
            ('bbh', _BB_HOT_COLOR,  '--', 'BB hot'),
            ('bbc', _BB_WARM_COLOR, ':',  'BB warm'),
        ):
            if key in rad:
                data = normalize(rad[key]) if normalized else rad[key]
                self._ax.plot(wn, data, color=color, ls=ls, lw=1.2, label=label)

        self._ax.legend(fontsize=9, loc='upper left')

    def _redraw_emissivity(self, normalized: bool) -> None:
        """Draw calibrated emissivity spectra from ``_emcal_result``."""
        r     = self._emcal_result
        wn    = r.get('xaxis', np.array([]))
        emiss = r.get('emiss', {})

        if wn.size:
            self._ax.set_xlim(wn.max(), wn.min())
        self._ax.set_ylabel('Emissivity (normalized)' if normalized else 'Emissivity')

        all_sample_names = r.get('label', [])
        names_to_plot = self._names_for_current_mode(all_sample_names)

        for n in names_to_plot:
            if n not in emiss:
                continue
            data = normalize(emiss[n]) if normalized else emiss[n]
            self._ax.plot(wn, data, label=n)

        self._ax.legend(fontsize=9, loc='upper left')

    def _redraw_ratio(self, normalized: bool, quantity: str) -> None:
        """
        Draw transmittance, reflectance, absorbance, or optical depth from
        the pre-computed ``_transmit_result`` or ``_reflect_result`` dict.

        All arithmetic is performed by :func:`transmit` / :func:`reflect`;
        this method only reads the stored arrays and plots them.

        Parameters
        ----------
        quantity : str
            One of ``'transmittance'``, ``'reflectance'``, ``'absorbance'``,
            ``'od'``.
        """
        _YLABEL = {
            'transmittance': 'Transmittance',
            'reflectance':   'Reflectance',
            'absorbance':    'Absorbance (−log₁₀ T)',
            'od':            'Optical Depth (−ln T)',
        }
        ylabel = _YLABEL.get(quantity, quantity)
        self._ax.set_ylabel(ylabel + ' (normalized)' if normalized else ylabel)

        # Route to the correct pre-computed result and output key.
        if quantity in ('transmittance', 'absorbance', 'od'):
            result   = self._tracal_result
            data_key = {'transmittance': 'tra', 'absorbance': 'abs', 'od': 'od'}[quantity]
        else:  # 'reflectance'
            result   = self._refcal_result
            data_key = 'ref'

        if result is None:
            self._ax.set_xlim(2000, 200)
            return

        wn      = result.get('wn', np.array([]))
        spectra = result.get(data_key, {})
        labels  = result.get('header', {}).get('sample_labels', [])

        if wn.size:
            self._ax.set_xlim(wn.max(), wn.min())

        names_to_plot = self._names_for_current_mode(labels)

        for name in names_to_plot:
            if name not in spectra:
                continue
            data = normalize(spectra[name]) if normalized else spectra[name]
            self._ax.plot(wn, data, label=name)

        self._ax.legend(fontsize=9, loc='upper left')

    def _redraw_kirchhoff(self, normalized: bool) -> None:
        """
        Draw the Kirchhoff complement (1 minus the primary quantity) for the
        current mode from the pre-computed result dict.

        * Emission     → 1 − Emissivity  (from ``_emcal_result['emiss']``)
        * Transmission → 1 − Transmittance  (from ``_transmit_result['tra']``)
        * Reflectance  → 1 − Reflectance  (from ``_reflect_result['ref']``)

        Parameters
        ----------
        normalized : bool
            When True, each spectrum is normalised to its peak before plotting.
        """
        mode = self._mode_var.get()
        if mode == 'Emission':
            r = self._emcal_result
            if r is None:
                return
            wn      = r.get('xaxis', np.array([]))
            spectra = r.get('emiss', {})
            labels  = r.get('label', [])
            ylabel  = '1−Emissivity (Kirchhoff)'
        elif mode == 'Transmission':
            r = self._tracal_result
            if r is None:
                return
            wn      = r.get('wn', np.array([]))
            spectra = r.get('tra', {})
            labels  = r.get('header', {}).get('sample_labels', [])
            ylabel  = '1−Transmittance (Kirchhoff)'
        else:  # Reflectance
            r = self._refcal_result
            if r is None:
                return
            wn      = r.get('wn', np.array([]))
            spectra = r.get('ref', {})
            labels  = r.get('header', {}).get('sample_labels', [])
            ylabel  = '1−Reflectance (Kirchhoff)'

        if wn.size:
            self._ax.set_xlim(wn.max(), wn.min())
        self._ax.set_ylabel(ylabel + ' (normalized)' if normalized else ylabel)

        names_to_plot = self._names_for_current_mode(labels)
        for name in names_to_plot:
            if name not in spectra:
                continue
            data = 1.0 - spectra[name]
            if normalized:
                data = normalize(data)
            self._ax.plot(wn, data, label=name)

        self._ax.legend(fontsize=9, loc='upper left')

    def _names_for_current_mode(self, all_names: list[str]) -> list[str]:
        """Return the subset of *all_names* to plot given stacked/single mode."""
        if self._plot_mode_var.get() == 'stacked':
            return all_names
        sel = self._spectra_lb.curselection()
        if sel:
            chosen = self._spectra_lb.get(sel[0])
            if chosen in all_names:
                return [chosen]
        return [all_names[-1]] if all_names else []

    def _refresh_plot(self) -> None:
        """Rebuild the plot from stored spectra (called on mode/scale toggle)."""
        self._redraw_plot()

    def _add_spectrum_to_display(
        self,
        spectrum: dict,
        name: str,
        mm_stats: dict | None = None,
        is_bb: bool = False,
    ) -> None:
        """
        Store a spectrum (with optional mm_stats) and update the listbox and plot.

        If *name* already exists in the store (e.g. a re-collect), the stored
        data is updated without adding a duplicate listbox entry.

        Parameters
        ----------
        spectrum : dict
            ``{'wn': np.ndarray, 'data': np.ndarray}``.
        name : str
            Sample name used as the listbox label and legend entry.
        mm_stats : dict or None
            Per-channel statistics dict from :meth:`_compute_mm_stats`.
        is_bb : bool
            True if this spectrum is a blackbody measurement.
        """
        if name not in self._spectra_data:
            self._spectra_lb.insert(tk.END, name)
            self._spectra_lb.selection_clear(0, tk.END)
            self._spectra_lb.selection_set(tk.END)
        self._spectra_data[name] = {
            **spectrum,
            'mm_stats': mm_stats or {},
            'is_bb':    is_bb,
            'is_bkg':   self._active_collection_is_bkg,
            'is_blank': self._active_collection_is_blank,
            'mode':     self._active_collection_mode,
        }
        # A T/R background collection stores the spectrum for transmit()/reflect().
        if self._active_collection_is_bkg:
            self._bkg_spectrum = spectrum
        self._check_processing_buttons()
        self._redraw_plot()
        if self._mm_mode_var.get() == 'sample':
            self._display_selected_spectrum_channels()

    def _on_color_scheme_change(self, _event=None) -> None:
        """Update the sample color cycle and redraw."""
        self._plot_colors = list(_COLOR_SCHEMES[self._color_scheme_var.get()])
        self._redraw_plot()

    # -----------------------------------------------------------------------
    # emcal
    # -----------------------------------------------------------------------

    _BB_HOT_TERMS  = {'bbhot', 'bbh', 'hotbb'}
    _BB_WARM_TERMS = {'bbcold', 'bbc', 'coldbb', 'bbwarm', 'bbw', 'warmbb'}

    @property
    def _has_sample_spectra(self) -> bool:
        """True if at least one non-BB, non-background, non-blank spectrum is in memory."""
        return any(
            not e.get('is_bb') and not e.get('is_bkg') and not e.get('is_blank')
            for e in self._spectra_data.values()
        )

    def _check_processing_buttons(self) -> None:
        """Enable the appropriate processing button for the current mode."""
        mode = self._mode_var.get()
        if mode == 'Emission':
            if self._save_fdir is None:
                self._btn_emcal.config(state='disabled')
                return
            bb_names = [n.lower() for n, e in self._spectra_data.items()
                        if e.get('is_bb')]
            has_hot  = any(any(t in n for t in self._BB_HOT_TERMS)  for n in bb_names)
            has_warm = any(any(t in n for t in self._BB_WARM_TERMS) for n in bb_names)
            self._btn_emcal.config(
                state='normal' if (has_hot and has_warm) else 'disabled')
            # Kirchhoff RB: enabled after emcal has produced a result.
            self._kirchhoff_rb.config(
                state='normal' if self._emcal_result is not None else 'disabled')
        elif mode == 'Transmission':
            # Process button: requires a background AND at least one sample.
            can_process = self._bkg_spectrum is not None and self._has_sample_spectra
            self._btn_tracal.config(state='normal' if can_process else 'disabled')
            # Display RBs: enabled only after transmit() has produced a result.
            dsp_state = 'normal' if self._tracal_result is not None else 'disabled'
            self._transmittance_rb.config(state=dsp_state)
            self._absorbance_rb.config(state=dsp_state)
            self._od_rb.config(state=dsp_state)
            self._kirchhoff_rb.config(state=dsp_state)
        elif mode == 'Reflectance':
            # Process button: requires a background AND at least one sample.
            can_process = self._bkg_spectrum is not None and self._has_sample_spectra
            self._btn_refcal.config(state='normal' if can_process else 'disabled')
            # Display RBs: enabled only after reflect() has produced a result.
            dsp_state = 'normal' if self._refcal_result is not None else 'disabled'
            self._reflectance_rb.config(state=dsp_state)
            self._kirchhoff_rb.config(state=dsp_state)

    def _on_emcal(self) -> None:
        """Open the options dialog, then run emcal() in a background thread."""
        if self._save_fdir is None:
            return
        dlg = EmcalOptionsDialog(self, self._emcal_opts)
        if dlg.result is None:
            return
        self._emcal_opts.update(dlg.result)
        logging.info(
            "emcal() started — method=%s, lab=%s, wn_range=%s",
            self._emcal_opts.get('method'),
            self._emcal_opts.get('lab'),
            self._emcal_opts.get('wn_range'),
        )
        self._btn_emcal.config(state='disabled')
        self._status_var.set('Running emcal()…')
        threading.Thread(
            target=self._run_emcal,
            args=(str(self._save_fdir), dict(self._emcal_opts)),
            daemon=True,
        ).start()

    def _run_emcal(self, fdir: str, opts: dict) -> None:
        """Background: call emcal() and post the result to the main thread."""

        def _bb_provider() -> tuple[float, float]:
            """Show _BBTempsDialog on the main thread; block until user responds."""
            holder: list = [None]
            event  = threading.Event()

            def _show() -> None:
                dlg = _BBTempsDialog(self)
                holder[0] = dlg.result
                event.set()

            self.after(0, _show)
            event.wait()
            if holder[0] is None:
                raise MissingTempsError("BB temperature entry cancelled.")
            return holder[0]

        try:
            result = emcal(
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
                save                = True,
                ow                  = True,
                on_missing_bb_temps = _bb_provider,
            )
            self.after(0, lambda r=result: self._on_emcal_result(r))
        except MissingTempsError as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: self._status_var.set(f'emcal() aborted: {m}'))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: (
                messagebox.showerror('emcal() failed', m),
                self._status_var.set('emcal() failed — see log'),
            ))
            logging.exception("emcal() failed")
        finally:
            self.after(0, self._check_processing_buttons)

    def _on_emcal_result(self, result: dict) -> None:
        """Main thread: store emcal result, enable Radiance/Emissivity, switch to Emissivity."""
        self._emcal_result = result
        self._radiance_rb.config(state='normal')
        self._emissivity_rb.config(state='normal')
        self._kirchhoff_rb.config(state='normal')
        n = len(result.get('label', []))
        self._status_var.set(f'emcal() complete — {n} sample(s)')
        logging.info("emcal() complete — %d samples", n)
        self._display_var.set('emissivity')
        self._redraw_plot()

    # -----------------------------------------------------------------------
    # Multimeter helpers
    # -----------------------------------------------------------------------

    def update_channel(self, channel: int, value: float | None) -> None:
        """
        Update the displayed reading for one multimeter channel.

        Parameters
        ----------
        channel : int
            Channel number (101–107).
        value : float or None
            Measured value; pass None or NaN to reset the field to '—'.
        """
        if channel not in self._channel_vars:
            return
        blank = value is None or (isinstance(value, float) and np.isnan(value))
        self._channel_vars[channel].set('—' if blank else f'{value:.2f}')

    def update_all_channels(self, readings: dict) -> None:
        """
        Bulk-update all multimeter channel displays.

        Parameters
        ----------
        readings : dict[int, float]
            Mapping of channel number → measured value.
        """
        for ch, val in readings.items():
            self.update_channel(ch, val)
        if 101 in readings or 102 in readings:
            self._update_bb_temp()

    def _update_bb_temp(self) -> None:
        """Recompute and display the estimated BB target temperature from channels 101 & 102."""
        r1 = self._live_readings.get(101)
        r2 = self._live_readings.get(102)
        if r1 is None or r2 is None:
            self._bb_temp_var.set('—')
            return
        try:
            temp_c = r2t_nau(r1, r2) - 273.15
            self._bb_temp_var.set(f'{temp_c:.2f}')
        except Exception:
            self._bb_temp_var.set('err')

    # -----------------------------------------------------------------------
    # Progress helpers
    # -----------------------------------------------------------------------

    def set_progress(
        self, value: float, status: str = '', n_scans: 'int | None' = None
    ) -> None:
        """
        Update the progress bar, scan counter, and optional status message.

        Parameters
        ----------
        value : float
            Completion percentage, 0–100.
        status : str
            Short status string; unchanged if empty.
        n_scans : int or None
            Number of scans completed so far.  When supplied, the label
            shows ``"n / total scans"`` using the target from
            ``_exp_params['num_scans']``; when ``None`` the label is cleared.
        """
        self._progress_var.set(value)
        if n_scans is not None:
            try:
                total = int(self._exp_params.get('num_scans', 0))
            except (ValueError, TypeError):
                total = 0
            scans_str = f'{n_scans} / {total}' if total > 0 else str(n_scans)
            self._scan_label.config(text=f'{scans_str} scans  ({value:.0f} %)')
        else:
            self._scan_label.config(text='')
        if status:
            self._status_var.set(status)

    # -----------------------------------------------------------------------
    # Folder management
    # -----------------------------------------------------------------------

    def _has_session_data(self) -> bool:
        """Return True if any spectra or processed results are held in memory."""
        return bool(self._spectra_data) or any(
            r is not None
            for r in (self._emcal_result, self._tracal_result, self._refcal_result)
        )

    def _clear_session_data(self) -> None:
        """Wipe all in-memory spectra and processed results and reset the plot."""
        self._spectra_data.clear()
        self._spectra_lb.delete(0, tk.END)
        self._bkg_spectrum    = None
        self._emcal_result    = None
        self._tracal_result = None
        self._refcal_result  = None
        self._display_var.set('sbm')
        self._redraw_plot()
        self._check_processing_buttons()

    def _on_browse_folder(self) -> None:
        """Open a directory chooser, update the save path, and init the spectrum count."""
        if self._has_session_data():
            n = len(self._spectra_data)
            noun = 'spectrum' if n == 1 else 'spectra'
            if not messagebox.askyesno(
                'Change folder?',
                f'You have {n} {noun} in memory.\n\n'
                'Switching to a new folder will clear all spectra and '
                'processed results.\n\nProceed?',
                icon='warning',
                default='no',
            ):
                return
        # Open the dialog at the last-used folder if available, otherwise the
        # user's home directory.  Avoid opening directly into the Nextcloud data
        # folder as its filesystem filter driver can make the dialog very slow
        # to render; the user can navigate there manually if needed.
        _init = self._save_fdir or Path.home()
        chosen = filedialog.askdirectory(
            title='Select save folder',
            initialdir=str(_init),
        )
        if not chosen:
            return
        self._save_fdir = Path(chosen)
        self._save_fdir_var.set(str(self._save_fdir))
        logging.info("Save folder set: %s", self._save_fdir)

        # Ensure the directory exists on disk — the Nextcloud client can move or
        # delete a local folder between the time the user selects it in the dialog
        # and the time we first write to it (sync conflicts, remote deletion, etc.).
        try:
            self._save_fdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("Could not create save folder: %s", exc)

        # If a measurement log already exists in this folder, resume its count.
        csv_path = self._save_fdir / f'{self._save_fdir.name}-measurement-info.csv'
        if csv_path.exists():
            try:
                with open(csv_path, 'r') as fh:
                    # Subtract 1 for the header row; floor at 0.
                    self._spectrum_count = max(0, sum(1 for _ in fh) - 1)
                logging.info("Resuming session; spectrum count: %d",
                             self._spectrum_count)
            except Exception as exc:
                logging.warning("Could not read existing CSV count: %s", exc)
                self._spectrum_count = 0
        else:
            self._spectrum_count = 0

        # ── Reset in-memory state for the new folder ─────────────────────────
        self._clear_session_data()
        self._status_var.set('Ready')

        self._refresh_buttons()
        self._check_processing_buttons()

    def _refresh_buttons(self) -> None:
        """Enable/disable collection buttons and show the correct mode-specific widgets."""
        mode  = self._mode_var.get()
        state = 'normal' if self._save_fdir is not None else 'disabled'

        self._btn_collect_sample.config(state=state)

        # ── Repack the entire left side unconditionally to guarantee order ──
        # All collection buttons, the name entry, the separator, and the
        # processing sub-frame are always unpacked then repacked so that a
        # mode switch never leaves the name entry stranded to the left.
        for w in (self._btn_collect_bb, self._btn_collect_bkg,
                  self._btn_collect_blank, self._btn_collect_sample,
                  self._name_label, self._sample_name_entry,
                  self._coll_proc_sep, self._proc_frame):
            w.pack_forget()

        if mode == 'Emission':
            self._btn_collect_bb.pack(side=tk.LEFT, padx=(0, 4))
            self._btn_collect_bb.config(state=state)
        else:
            self._btn_collect_bkg.pack(side=tk.LEFT, padx=(0, 4))
            self._btn_collect_blank.pack(side=tk.LEFT, padx=(0, 4))
            self._btn_collect_bkg.config(state=state)
            self._btn_collect_blank.config(state=state)

        self._btn_collect_sample.pack(side=tk.LEFT, padx=(0, 4))
        self._name_label.pack(side=tk.LEFT, padx=(6, 2))
        self._sample_name_entry.pack(side=tk.LEFT, padx=(0, 4))
        self._coll_proc_sep.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        self._proc_frame.pack(side=tk.LEFT, padx=(0, 4))

        # Purge delay spinbox: visible only in T/R modes (packed after proc_frame).
        self._purge_frame.pack_forget()
        if mode != 'Emission':
            self._purge_frame.pack(side=tk.LEFT, padx=(4, 0))

        # Switch processing button — one visible at a time inside _proc_frame.
        for btn, target_mode in (
            (self._btn_emcal,    'Emission'),
            (self._btn_tracal, 'Transmission'),
            (self._btn_refcal,  'Reflectance'),
        ):
            if mode == target_mode:
                if not btn.winfo_ismapped():
                    btn.pack(side=tk.LEFT)
            else:
                btn.pack_forget()

        # ── Display mode radiobuttons: show only those for the current mode ─
        # Update Kirchhoff label to match the active mode's primary quantity.
        if mode == 'Emission':
            self._kirchhoff_rb.config(text='1−E (Kirchhoff)')
            _visible_dsp = (self._radiance_rb, self._emissivity_rb, self._kirchhoff_rb)
        elif mode == 'Transmission':
            self._kirchhoff_rb.config(text='1−T (Kirchhoff)')
            _visible_dsp = (self._transmittance_rb, self._absorbance_rb,
                            self._od_rb, self._kirchhoff_rb)
        else:
            self._kirchhoff_rb.config(text='1−R (Kirchhoff)')
            _visible_dsp = (self._reflectance_rb, self._kirchhoff_rb)

        # Unique set of all possible RBs for hide-before-show.
        _all_dsp = (self._radiance_rb, self._emissivity_rb,
                    self._transmittance_rb, self._absorbance_rb, self._od_rb,
                    self._reflectance_rb, self._kirchhoff_rb)

        for rb in _all_dsp:
            rb.pack_forget()
        prev = self._sbm_rb
        for rb in _visible_dsp:
            rb.pack(side=tk.LEFT, padx=(0, 2), after=prev)
            prev = rb

        self._check_processing_buttons()

    # -----------------------------------------------------------------------
    # Mode switching
    # -----------------------------------------------------------------------

    def _on_mode_change(self) -> None:
        """
        Fired when the mode radiobutton changes.

        Warns the user if spectra or results are in memory (they will be
        cleared), reverts the selection if cancelled.  On confirmation,
        clears session data, updates the OMNIC experiment file list,
        auto-loads the new experiment if connected, and refreshes buttons.
        """
        mode = self._mode_var.get()
        if mode == self._last_mode:
            return   # spurious fire — nothing actually changed

        if self._has_session_data():
            n = len(self._spectra_data)
            noun = 'spectrum' if n == 1 else 'spectra'
            if not messagebox.askyesno(
                'Switch mode?',
                f'You have {n} {noun} in memory.\n\n'
                f'Switching from {self._last_mode} to {mode} mode will clear '
                'all spectra and processed results.\n\nProceed?',
                icon='warning',
                default='no',
            ):
                logging.info(
                    "Mode switch to %s cancelled (%d %s in memory)",
                    mode, n, noun)
                # Revert the radiobutton without re-triggering this handler.
                self._mode_var.set(self._last_mode)
                return

        logging.info("Mode switched: %s → %s", self._last_mode, mode)
        self._last_mode = mode
        self._clear_session_data()

        self._exp_files = _scan_exp_files(mode)
        default = _MODE_EXP_DEFAULTS.get(mode, DEFAULT_EXP_FILENAME)
        self._exp_file_var.set(
            default if default in self._exp_files else self._exp_files[0])
        self._cb_exp.config(values=self._exp_files)
        if self._spec.connected:
            self._spec_auto_load_params()
        self._refresh_buttons()

    # -----------------------------------------------------------------------
    # OMNIC parameter loading
    # -----------------------------------------------------------------------

    def _on_exp_file_change(self, _event=None) -> None:
        """Auto-load the newly selected .exp file if the spectrometer is connected."""
        exp_file = self._exp_file_var.get()
        logging.info("Experiment file selected: %s", exp_file)
        if self._spec.connected:
            self._spec_auto_load_params()

    def _on_show_params(self) -> None:
        """Open the experiment parameters dialog."""
        ExperimentParamsDialog(self, self._exp_params)

    # -----------------------------------------------------------------------
    # Multimeter refresh (one-shot)
    # -----------------------------------------------------------------------

    def _on_refresh(self) -> None:
        """Trigger a one-shot multimeter read in a background thread."""
        if not self._mm.connected:
            messagebox.showwarning('Not connected', 'Multimeter is not connected.')
            return
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _refresh_worker(self) -> None:
        """Background: read all channels; only update display when in Live mode."""
        try:
            readings = self._mm.read_all_channels(list(_PANEL_ORDER))
            self.after(0, lambda r=readings: self._on_refresh_result(r))
        except Exception as exc:
            logging.error("One-shot refresh failed: %s", exc)
            self.after(0, lambda msg=str(exc):
                       messagebox.showerror('Multimeter error', msg))

    def _on_refresh_result(self, readings: dict[int, float]) -> None:
        """Main thread: store live snapshot; only push to display in Live mode."""
        self._live_readings = readings
        if self._mm_mode_var.get() == 'live':
            self.update_all_channels(readings)

    # -----------------------------------------------------------------------
    # Spectra list
    # -----------------------------------------------------------------------

    def _on_spectra_select(self, _event=None) -> None:
        """Plot the selected listbox spectrum and update Sample-mode MM display."""
        sel = self._spectra_lb.curselection()
        if not sel:
            return
        self._redraw_plot()
        if self._mm_mode_var.get() == 'sample':
            self._display_selected_spectrum_channels()

    def _selected_spectrum_name(self) -> str | None:
        """Return the name of the currently selected listbox entry, or None."""
        sel = self._spectra_lb.curselection()
        return self._spectra_lb.get(sel[0]) if sel else None

    def _remove_spectrum_from_list(self, name: str) -> None:
        """Remove *name* from the in-memory store and the listbox, then redraw.

        Any processed result (emcal / transmit / reflect) that included this
        spectrum is cleared — it is now stale and must be re-run.
        """
        self._spectra_data.pop(name, None)
        items = list(self._spectra_lb.get(0, tk.END))
        if name in items:
            self._spectra_lb.delete(items.index(name))

        # Invalidate processed results that contained this spectrum.
        reset_display = False
        if self._emcal_result is not None:
            if name in self._emcal_result.get('label', []):
                self._emcal_result = None
                self._radiance_rb.config(state='disabled')
                self._emissivity_rb.config(state='disabled')
                self._kirchhoff_rb.config(state='disabled')
                reset_display = True
        if self._tracal_result is not None:
            if name in self._tracal_result.get('header', {}).get('sample_labels', []):
                self._tracal_result = None
                reset_display = True
        if self._refcal_result is not None:
            if name in self._refcal_result.get('header', {}).get('sample_labels', []):
                self._refcal_result = None
                reset_display = True
        if reset_display:
            self._display_var.set('sbm')

        self._check_processing_buttons()
        self._redraw_plot()

    def _on_delete_spectrum(self) -> None:
        """Delete the selected spectrum: remove from list, delete CSV, remove log row."""
        name = self._selected_spectrum_name()
        if name is None:
            return

        confirmed = messagebox.askyesno(
            'Confirm Delete',
            f'Permanently delete "{name}"?\n\n'
            'This will:\n'
            f'  • Remove it from the plot\n'
            f'  • Delete {name}.CSV from the save folder\n'
            f'  • Remove its row from the measurement-info CSV\n\n'
            'This cannot be undone.',
            icon='warning',
        )
        if not confirmed:
            return

        logging.info("Deleting spectrum: %s", name)
        errors: list[str] = []

        # Delete the spectrum CSV file.
        if self._save_fdir is not None:
            csv_file = self._save_fdir / f'{name}.CSV'
            try:
                if csv_file.exists():
                    csv_file.unlink()
                    logging.info("Deleted spectrum file: %s", csv_file)
                else:
                    logging.warning("Spectrum file not found, skipping delete: %s", csv_file)
            except Exception as exc:
                errors.append(f'Could not delete {csv_file.name}: {exc}')
                logging.error("Failed to delete %s: %s", csv_file, exc)

            # Remove the row from measurement-info CSV.
            info_path = self._save_fdir / f'{self._save_fdir.name}-measurement-info.csv'
            if info_path.exists():
                try:
                    with open(info_path, 'r', newline='') as fh:
                        reader = csv.DictReader(fh)
                        fieldnames = reader.fieldnames or []
                        rows = [r for r in reader if r.get('sample_name') != name]
                    with open(info_path, 'w', newline='') as fh:
                        writer = csv.DictWriter(fh, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    logging.info("Removed '%s' from measurement-info", name)
                except Exception as exc:
                    errors.append(f'Could not update measurement-info: {exc}')
                    logging.error("Failed to update measurement-info: %s", exc)

        self._remove_spectrum_from_list(name)

        if errors:
            messagebox.showwarning(
                'Delete incomplete',
                'Spectrum removed from list, but:\n\n' + '\n'.join(errors),
            )

    def _on_delete_all_spectra(self) -> None:
        """Delete all collected spectra: clear list, delete CSV files, remove log rows."""
        n = len(self._spectra_data)
        if n == 0:
            return
        noun = 'spectrum' if n == 1 else 'spectra'
        if not messagebox.askyesno(
            'Confirm Delete All',
            f'Permanently delete all {n} {noun}?\n\n'
            'This will:\n'
            f'  • Delete {n} CSV file(s) from the save folder\n'
            '  • Remove their rows from the measurement-info CSV\n'
            '  • Clear all spectra from the plot\n\n'
            'This cannot be undone.',
            icon='warning',
            default='no',
        ):
            return

        names   = list(self._spectra_data.keys())
        logging.info("Deleting all spectra — %d item(s): %s", len(names), names)
        errors: list[str] = []

        if self._save_fdir is not None:
            # Delete each spectrum CSV file.
            for name in names:
                csv_file = self._save_fdir / f'{name}.CSV'
                try:
                    if csv_file.exists():
                        csv_file.unlink()
                        logging.info("Deleted spectrum file: %s", csv_file)
                    else:
                        logging.warning("Spectrum file not found, skipping: %s", csv_file)
                except Exception as exc:
                    errors.append(f'Could not delete {csv_file.name}: {exc}')
                    logging.error("Failed to delete %s: %s", csv_file, exc)

            # Remove all their rows from measurement-info in one pass.
            info_path = self._save_fdir / f'{self._save_fdir.name}-measurement-info.csv'
            if info_path.exists():
                name_set = set(names)
                try:
                    with open(info_path, 'r', newline='') as fh:
                        reader    = csv.DictReader(fh)
                        fieldnames = reader.fieldnames or []
                        rows      = [r for r in reader
                                     if r.get('sample_name') not in name_set]
                    with open(info_path, 'w', newline='') as fh:
                        writer = csv.DictWriter(fh, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(rows)
                    logging.info("Removed %d spectra from measurement-info", len(names))
                except Exception as exc:
                    errors.append(f'Could not update measurement-info: {exc}')
                    logging.error("Failed to update measurement-info: %s", exc)

        self._clear_session_data()

        if errors:
            messagebox.showwarning(
                'Delete incomplete',
                'Spectra cleared from list, but:\n\n' + '\n'.join(errors),
            )

    # -----------------------------------------------------------------------
    # Collection
    # -----------------------------------------------------------------------

    def _on_collect_sample(self) -> None:
        """Validate the sample name field and start a sample collection."""
        name = self._sample_name_entry.get().strip()
        if not name:
            messagebox.showwarning(
                'No name', 'Enter a sample name in the Name field before collecting.')
            return
        self._run_collection(name, is_bb=False)

    def _on_collect_bb(self) -> None:
        """Open BB confirmation dialog; start collection only if user confirms."""
        dlg = _BBCollectDialog(self, self._bb_type_var, self._bb_temp_var)
        if dlg.result is None:
            return   # user cancelled
        if dlg.result == 'bbgeneric':
            name = self._next_auto_name('bb')
        else:
            name = dlg.result   # 'bbwarm' or 'bbhot'
        self._run_collection(name, is_bb=True)

    def _next_auto_name(self, prefix: str) -> str:
        """Return the next unused ``prefix###`` name (e.g. ``bkg003``).

        Scans existing entries in :attr:`_spectra_data` for names that match
        ``<prefix><three-digit number>`` (case-insensitive) and returns the
        lowest unused candidate starting from ``001``.
        """
        p = prefix.lower()
        used: set[str] = set()
        for n in self._spectra_data:
            nl = n.lower()
            suffix = nl[len(p):]
            if nl.startswith(p) and len(suffix) == 3 and suffix.isdigit():
                used.add(nl)
        for i in range(1, 1000):
            candidate = f'{prefix}{i:03d}'
            if candidate.lower() not in used:
                return candidate
        return f'{prefix}999'   # extremely unlikely fallback

    def _on_collect_bkg(self) -> None:
        """Guard against accidental bkg collection, then auto-name and start."""
        name_text = self._sample_name_entry.get().strip()
        if name_text:
            if not messagebox.askyesno(
                'Collect Background?',
                f'A sample name ("{name_text}") is entered in the Name field.\n\n'
                'Are you sure you want to collect a Background — not a Sample?',
                icon='warning',
                default='no',
            ):
                return
        self._run_collection(self._next_auto_name('bkg'), is_bb=False, is_bkg=True)

    def _on_collect_blank(self) -> None:
        """Guard against accidental blank collection, then auto-name and start."""
        name_text = self._sample_name_entry.get().strip()
        if name_text:
            if not messagebox.askyesno(
                'Collect Blank?',
                f'A sample name ("{name_text}") is entered in the Name field.\n\n'
                'Are you sure you want to collect a Blank — not a Sample?',
                icon='warning',
                default='no',
            ):
                return
        self._run_collection(self._next_auto_name('blank'), is_bb=False, is_blank=True)

    def _warn_mm_disconnected(self, is_bb: bool) -> bool:
        """
        Show a warning dialog when the multimeter is not connected in Emission mode.

        Parameters
        ----------
        is_bb : bool
            True for a blackbody collection (stronger warning, default=no);
            False for a sample collection (softer warning, default=yes).

        Returns
        -------
        bool
            True if the user chose to proceed, False to abort the collection.
        """
        if is_bb:
            return messagebox.askyesno(
                'Multimeter not connected',
                'The multimeter is not connected.\n\n'
                'BB temperature will NOT be recorded. '
                'You will be prompted to enter temperatures manually '
                'when running emcal().\n\n'
                'Continue with BB collection anyway?',
                icon='warning',
                default='no',
            )
        return messagebox.askyesno(
            'Multimeter not connected',
            'The multimeter is not connected.\n\n'
            'Sample temperature will NOT be recorded for this spectrum.\n\n'
            'Continue with sample collection anyway?',
            icon='warning',
            default='yes',
        )

    def _run_collection(
        self, name: str, is_bb: bool, is_blank: bool = False, is_bkg: bool = False,
    ) -> None:
        """Gate-check then start the collection state machine."""
        if self._collection_active:
            messagebox.showwarning('Busy', 'A collection is already in progress.')
            return
        if self._save_fdir is None:
            messagebox.showerror(
                'No folder',
                'Select a save folder before collecting.\n'
                'Data cannot be collected without a destination.')
            return
        if not self._spec.connected:
            messagebox.showerror(
                'Not connected',
                'Spectrometer is not connected.\n'
                'Connect to OMNIC before collecting.')
            return
        # Warn if multimeter is disconnected in Emission mode.
        if self._mode_var.get() == 'Emission' and not self._mm.connected:
            if not self._warn_mm_disconnected(is_bb):
                return
        if name in self._spectra_data:
            overwrite = messagebox.askyesno(
                'Duplicate name',
                f'A spectrum named "{name}" already exists.\n\n'
                'Overwrite it (replaces the plot entry, the .CSV file, and '
                'the measurement-info row)?\n\n'
                'Choose "No" to cancel and enter a different name.',
                icon='warning',
                default='no',
            )
            if not overwrite:
                return
            logging.info("Overwriting existing spectrum: %s", name)
            self._collection_overwrite = True
        else:
            self._collection_overwrite = False
        mode = self._mode_var.get()
        kind = ('BB' if is_bb else 'blank' if is_blank else 'bkg' if is_bkg else 'sample')
        logging.info("Collection started: %s  [mode=%s, type=%s]", name, mode, kind)
        self._collection_active          = True
        self._active_collection_mode     = self._mode_var.get()
        self._active_collection_is_blank = is_blank
        self._active_collection_is_bkg   = is_bkg
        self._refresh_buttons()

        if self._active_collection_mode == 'Emission':
            # Pause live poll; measurement sampling thread takes over during collection.
            self._stop_live_poll()
            # Reset sample accumulators for this collection.
            self._mm_measurement_samples    = {ch: [] for ch in _PANEL_ORDER}
            self._mm_measurement_timestamps = []
            self._stop_mm_measurement.clear()
            self._status_var.set('Reading multimeter…')
            threading.Thread(
                target=self._coll_read_mm,
                args=(name, is_bb),
                daemon=True,
            ).start()
        else:
            # Transmission / Reflectance: no MM sampling.
            # Show the purge countdown before starting the DDE command.
            self.after(0, lambda: self._coll_purge_delay(name, is_bb))

    # -- Step 1a: purge equilibration countdown (T/R, main thread) -----------

    def _on_purge_toggle(self) -> None:
        """Enable or disable the purge delay spinbox to match the checkbutton state."""
        self._purge_spinbox.config(
            state='normal' if self._purge_enabled_var.get() else 'disabled')

    def _coll_purge_delay(self, name: str, is_bb: bool) -> None:
        """Main thread: show purge countdown dialog, then start collection.

        Called only for T/R modes.  Shutters are always open (manual mode set
        in the ``.exp`` file), so this delay purely allows residual H₂O/CO₂ to
        clear from the sample compartment after it was opened.
        If the purge delay is disabled via the checkbutton, collection starts
        immediately without a countdown.
        """
        if not self._purge_enabled_var.get():
            self._coll_start_collect(name, is_bb)
            return
        self._status_var.set('Waiting for purge equilibration…')
        _PurgeCountdownDialog(
            parent=self,
            delay_s=self._purge_delay_var.get(),
            on_proceed=lambda: self._coll_start_collect(name, is_bb),
        )

    # -- Step 1b: pre-collection MM read (background, Emission only) ----------

    def _coll_read_mm(self, name: str, is_bb: bool) -> None:
        """Background: take first (pre-collection bracket) reading, hand off to main."""
        if self._mm.connected:
            try:
                ts       = datetime.now()
                readings = self._mm.read_all_channels(list(_PANEL_ORDER))
                self.after(0, lambda r=readings, t=ts: self._on_measurement_sample(r, t))
            except Exception as exc:
                logging.warning("Pre-collection MM read failed: %s", exc)
                self.after(0, lambda: self._status_var.set(
                    '⚠ MM pre-collection read failed — temperature bracket missing'))
        self.after(0, lambda: self._coll_start_collect(name, is_bb))

    # -- Step 2: send CollectSample (main thread) ----------------------------

    def _coll_start_collect(self, name: str, is_bb: bool, attempt: int = 1) -> None:
        """Main thread: send CollectSample DDE command; retry on failure.

        Sends ``[CollectSample "name" Auto Polling]`` for all modes — no OMNIC
        prompts, no collection window.  T/R purge equilibration is handled
        before this step by :meth:`_coll_purge_delay`.
        """
        max_retries = COLLECT_MAX_RETRIES
        self._status_var.set(f'Collecting… (attempt {attempt}/{max_retries})')
        try:
            self._spec.start_collect(name)
            start_dtime = datetime.now()
            self._coll_start_time = start_dtime
            self.after(1000, self._elapsed_tick)
            # Start high-frequency measurement sampling only in Emission mode.
            if self._active_collection_mode == 'Emission':
                self._stop_mm_measurement.clear()
                self._mm_sample_thread = threading.Thread(
                    target=self._measurement_sample_loop, daemon=True)
                self._mm_sample_thread.start()
            self.after(2000, lambda: self._coll_poll(name, is_bb, start_dtime))
        except Exception as exc:
            if attempt < max_retries:
                logging.warning(
                    "CollectSample attempt %d/%d failed; retrying in %ds",
                    attempt, max_retries, COLLECT_RETRY_DELAY_S,
                )
                self.after(
                    COLLECT_RETRY_DELAY_S * 1000,
                    lambda: self._coll_start_collect(name, is_bb, attempt + 1),
                )
            else:
                logging.error(
                    "CollectSample failed after %d attempts: %s",
                    max_retries, exc,
                )
                messagebox.showerror('Collection failed', str(exc))
                self._coll_abort('Error')

    # -- Step 2b: elapsed-time ticker (main thread) --------------------------

    def _elapsed_tick(self) -> None:
        """Main thread: update the status bar with elapsed collection time.

        Reschedules itself every second while a collection is active.
        Stops automatically when _collection_active is cleared by _coll_abort.
        """
        if not self._collection_active or self._coll_start_time is None:
            return
        elapsed = int((datetime.now() - self._coll_start_time).total_seconds())
        m, s = divmod(elapsed, 60)
        self._status_var.set(f'Collecting…  {m}:{s:02d}')
        self.after(1000, self._elapsed_tick)

    # -- Step 3: poll OMNIC until complete (main thread) ---------------------

    # Maximum consecutive poll errors before aborting.  A small number of
    # transient NACKs can occur during the hardware shutter/purge sequence.
    _POLL_ERROR_TOLERANCE = 5

    def _coll_poll(
        self,
        name: str,
        is_bb: bool,
        start_dtime: datetime,
        poll_errors: int = 0,
    ) -> None:
        """Main thread: poll Collect Status every 2 s until pct == 100.

        Consecutive poll errors are tolerated up to ``_POLL_ERROR_TOLERANCE``
        before the collection is aborted.  Transient NACKs can occur while
        OMNIC runs its hardware shutter/purge sequence at the start of T/R
        collections.
        """
        try:
            n_scans, pct = self._spec.poll_collect_status()
            self.set_progress(float(pct), 'Collecting…', n_scans=n_scans)
            if pct == 100:
                end_dtime = datetime.now()
                self._coll_finish_dde(name, is_bb, start_dtime, end_dtime)
            else:
                self.after(2000,
                           lambda: self._coll_poll(name, is_bb, start_dtime))
        except Exception as exc:
            poll_errors += 1
            if poll_errors <= self._POLL_ERROR_TOLERANCE:
                logging.warning(
                    "Collect poll error %d/%d (may be transient): %s",
                    poll_errors, self._POLL_ERROR_TOLERANCE, exc,
                )
                self.after(
                    2000,
                    lambda e=poll_errors: self._coll_poll(
                        name, is_bb, start_dtime, poll_errors=e),
                )
            else:
                logging.error(
                    "Collect poll failed after %d consecutive errors: %s",
                    poll_errors, exc,
                )
                messagebox.showerror('Collection failed', str(exc))
                self._coll_abort('Error')

    # -- Step 4: DDE export (main thread) ------------------------------------

    def _coll_finish_dde(
        self,
        name: str,
        is_bb: bool,
        start_dtime: datetime,
        end_dtime: datetime,
    ) -> None:
        """Main thread: display and export spectrum, then kick off final MM read.

        ``[Display]`` is best-effort: in T/R modes OMNIC auto-displays the
        result spectrum the moment collection finishes, so the command may
        NACK.  We log the NACK as a warning and proceed to ``[Export]``.
        """
        # [Display] — non-fatal; OMNIC may already have auto-displayed the result.
        try:
            self._spec.display()
        except Exception as exc:
            logging.warning("[Display] NACK after collection (non-fatal): %s", exc)

        # [Export] — fatal; we need the CSV file on disk.
        # Re-create the directory in case it disappeared (Nextcloud sync, etc.).
        try:
            self._save_fdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("Could not ensure save folder exists: %s", exc)

        out_csv = str(self._save_fdir / f'{name}.CSV')
        try:
            self._spec.export_csv(out_csv)
            logging.info("Spectrum exported: %s", out_csv)
        except Exception as exc:
            logging.error("[Export] failed: %s", exc)
            messagebox.showerror('Collection failed', str(exc))
            self._coll_abort('Error')
            return

        threading.Thread(
            target=self._coll_final_mm,
            args=(name, is_bb, start_dtime, end_dtime, out_csv),
            daemon=True,
        ).start()

    # -- Step 5: post-collection MM read (background) ------------------------

    def _coll_final_mm(
        self,
        name: str,
        is_bb: bool,
        start_dtime: datetime,
        end_dtime: datetime,
        out_csv: str,
    ) -> None:
        """Background: stop sampling thread and take final MM reading (Emission only)."""
        if self._active_collection_mode == 'Emission':
            self._stop_mm_measurement.set()
            if self._mm.connected:
                try:
                    ts       = datetime.now()
                    readings = self._mm.read_all_channels(list(_PANEL_ORDER))
                    # Post the final sample BEFORE save_and_close so it's in stats.
                    self.after(0, lambda r=readings, t=ts:
                               self._on_measurement_sample(r, t))
                except Exception as exc:
                    logging.warning("Post-collection MM read failed: %s", exc)
                    self.after(0, lambda: self._status_var.set(
                        '⚠ MM post-collection read failed — temperature bracket missing'))
        self.after(
            0,
            lambda: self._coll_save_and_close(
                name, is_bb, start_dtime, end_dtime, out_csv),
        )

    # -- Step 6: compute stats, save CSV row, hide (main thread) -------------

    def _coll_save_and_close(
        self,
        name: str,
        is_bb: bool,
        start_dtime: datetime,
        end_dtime: datetime,
        out_csv: str,
    ) -> None:
        """Main thread: compute stats, hide in OMNIC, write log row, load plot."""
        if self._active_collection_mode == 'Emission':
            # All _on_measurement_sample after() callbacks have executed (FIFO queue),
            # so _mm_measurement_samples and timestamps are complete.
            # Flush every collection sample to the live log before computing stats.
            for i, ts in enumerate(self._mm_measurement_timestamps):
                sample = {
                    ch: self._mm_measurement_samples[ch][i]
                    for ch in _PANEL_ORDER
                    if i < len(self._mm_measurement_samples.get(ch, []))
                }
                self._append_live_log(
                    sample, ts, sample_name=name, is_bb=str(int(is_bb)))
            stats = self._compute_mm_stats()
        else:
            stats = {}
        try:
            self._spec.hide_selected()
            self._save_measurement_row(name, is_bb, start_dtime, end_dtime, stats)
            threading.Thread(
                target=self._coll_load_spectrum,
                args=(name, out_csv, stats, is_bb),
                daemon=True,
            ).start()
        except Exception as exc:
            logging.error("Collection save failed: %s", exc)
            messagebox.showerror('Collection failed', str(exc))
            self._coll_abort('Error')

    # -- Step 7: load spectrum into plot (background) ------------------------

    def _coll_load_spectrum(self, name: str, out_csv: str, stats: dict, is_bb: bool = False) -> None:
        """Background: read exported CSV and post spectrum + stats to main thread."""
        try:
            spectrum = readOMNIC(out_csv)
            self.after(0, lambda s=spectrum, n=name, st=stats, bb=is_bb:
                       self._add_spectrum_to_display(s, n, st, bb))
        except Exception as exc:
            logging.error("readOMNIC failed for %s: %s", out_csv, exc)
        finally:
            self.after(0, lambda: self._coll_abort('Ready'))

    def _coll_abort(self, status: str) -> None:
        """Main thread: reset collection state, re-enable buttons, resume live poll."""
        logging.info("Collection ended — %s", status)
        self._coll_start_time = None   # stops the elapsed-time ticker
        self.set_progress(0.0, status)
        self._collection_active = False
        self._refresh_buttons()
        # Live poll was paused only for Emission mode.
        if (self._active_collection_mode == 'Emission'
                and self._mm_mode_var.get() == 'live'
                and self._mm.connected):
            self._start_live_poll()

    # -- Measurement sampling helpers ----------------------------------------

    def _measurement_sample_loop(self) -> None:
        """Background: read all channels every MM_MEASUREMENT_INTERVAL_S during collection."""
        while not self._stop_mm_measurement.wait(MM_MEASUREMENT_INTERVAL_S):
            if not self._mm.connected:
                break
            ts       = datetime.now()
            readings = self._mm.read_all_channels(list(_PANEL_ORDER))
            if not self._mm.connected:
                break
            nan_chs = [ch for ch, v in readings.items() if np.isnan(v)]
            if nan_chs:
                logging.warning(
                    "Measurement sample: NaN on channels %s (transient read error)", nan_chs)
                self.after(0, lambda chs=nan_chs: self._status_var.set(
                    f'⚠ MM read warning — no data on channel(s) {chs}'))
            self.after(0, lambda r=readings, t=ts: self._on_measurement_sample(r, t))

    def _on_measurement_sample(
        self,
        readings: dict[int, float],
        ts: datetime | None = None,
    ) -> None:
        """Main thread: accumulate a measurement sample and refresh the display."""
        now = ts or datetime.now()
        self._mm_measurement_timestamps.append(now)
        for ch, val in readings.items():
            if ch in self._mm_measurement_samples:
                self._mm_measurement_samples[ch].append(val)
        self.update_all_channels(readings)
        if any(not np.isnan(v) for v in readings.values()):
            self._mm_last_updated_var.set(now.strftime('%H:%M:%S'))

    def _compute_mm_stats(self) -> dict[int, dict]:
        """Compute mean/std/min/max for each channel from accumulated samples.

        NaN values (from transient read failures) are excluded from all
        statistics; only valid readings contribute to mean, std, min, max, and
        the reported sample count ``n``.
        """
        stats: dict[int, dict] = {}
        for ch in _PANEL_ORDER:
            samples = self._mm_measurement_samples.get(ch, [])
            if samples:
                arr   = np.asarray(samples, dtype=float)
                valid = arr[~np.isnan(arr)]
                if len(valid) > 0:
                    stats[ch] = {
                        'mean': float(np.mean(valid)),
                        'std':  float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                        'min':  float(np.min(valid)),
                        'max':  float(np.max(valid)),
                        'n':    len(valid),
                    }
            else:
                stats[ch] = {'mean': '', 'std': '', 'min': '', 'max': '', 'n': 0}
        return stats

    def _save_measurement_row(
        self,
        name: str,
        is_bb: bool,
        start_dtime: datetime,
        end_dtime: datetime,
        stats: dict[int, dict],
    ) -> None:
        """
        Append one row to the session measurement log CSV.

        Legacy column names are preserved: ``channel_XXX`` holds the mean.
        New columns ``channel_XXX_std/min/max`` and timing fields are appended.
        """
        if self._save_fdir is None:
            return

        try:
            self._save_fdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logging.warning("Could not ensure save folder for measurement log: %s", exc)

        csv_path = self._save_fdir / f'{self._save_fdir.name}-measurement-info.csv'
        is_new   = not csv_path.exists()

        mode      = self._active_collection_mode
        n_samples = max(
            (stats[ch]['n'] for ch in _PANEL_ORDER if stats.get(ch)), default=0)
        mid_dtime = start_dtime + (end_dtime - start_dtime) / 2

        _EXP_PARAM_KEYS = [
            'exp_file', 'resolution', 'num_scans', 'apodization',
            'zero_fill', 'high_cutoff', 'low_cutoff', 'gain',
            'beamsplitter', 'velocity',
        ]
        _HEADERS = [
            'spectrum_number', 'sample_name', 'mode', 'is_bb', 'is_bkg', 'is_blank',
            'dtime', 'start_dtime', 'end_dtime', 'n_multi_samples',
        ]
        _HEADERS += [f'exp_{k}' if not k.startswith('exp_') else k
                     for k in _EXP_PARAM_KEYS]
        for ch in [101, 102, 103, 104, 105, 106, 107]:
            _HEADERS += [
                f'channel_{ch}',
                f'channel_{ch}_std',
                f'channel_{ch}_min',
                f'channel_{ch}_max',
            ]

        _RESISTANCE_CHS = {101, 102}

        def _s(ch: int, key: str) -> str:
            # No MM data for T/R collections.
            if mode != 'Emission':
                return ''
            if ch in _RESISTANCE_CHS and not is_bb:
                return ''
            v = stats.get(ch, {}).get(key, '')
            return f'{v:.4f}' if isinstance(v, float) else ''

        fmt = '%H:%M:%S'
        row = [
            self._spectrum_count,
            name,
            mode,
            int(is_bb),
            int(self._active_collection_is_bkg),
            int(self._active_collection_is_blank),
            mid_dtime.strftime(fmt),
            start_dtime.strftime(fmt),
            end_dtime.strftime(fmt),
            n_samples,
        ]
        row += [self._exp_params.get(k, '') for k in _EXP_PARAM_KEYS]
        for ch in [101, 102, 103, 104, 105, 106, 107]:
            row += [_s(ch, 'mean'), _s(ch, 'std'), _s(ch, 'min'), _s(ch, 'max')]

        if self._collection_overwrite and csv_path.exists():
            # Replace the existing row for this sample_name in-place.
            try:
                with open(csv_path, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    existing_fieldnames = reader.fieldnames or _HEADERS
                    existing_rows = list(reader)
                replaced = False
                for r in existing_rows:
                    if r.get('sample_name') == name:
                        for key, val in zip(_HEADERS, row):
                            r[key] = val
                        replaced = True
                        break
                with open(csv_path, 'w', newline='') as fh:
                    writer = csv.DictWriter(fh, fieldnames=existing_fieldnames)
                    writer.writeheader()
                    writer.writerows(existing_rows)
                if replaced:
                    logging.info("Measurement row overwritten for '%s' → %s", name, csv_path)
                    return  # don't increment spectrum_count for an overwrite
            except Exception as exc:
                logging.error("Overwrite of measurement-info failed, appending instead: %s", exc)

        with open(csv_path, 'a', newline='') as fh:
            writer = csv.writer(fh)
            if is_new:
                writer.writerow(_HEADERS)
            writer.writerow(row)

        logging.info("Measurement row %d written → %s",
                     self._spectrum_count, csv_path)
        self._spectrum_count += 1

    # -----------------------------------------------------------------------
    # Transmission / Reflectance processing stubs
    # -----------------------------------------------------------------------

    def _on_tracal(self) -> None:
        """Save transmit() results to disk in a background thread."""
        if self._save_fdir is None:
            return
        logging.info("tracal() started — folder: %s", self._save_fdir)
        self._btn_tracal.config(state='disabled')
        self._status_var.set('Running tracal()…')
        threading.Thread(
            target=self._run_tracal,
            args=(str(self._save_fdir),),
            daemon=True,
        ).start()

    def _run_tracal(self, fdir: str) -> None:
        """Background: call transmit(save=True) and post result to main thread."""
        try:
            result = tracal(fdir, save=True, ow=True)
            self.after(0, lambda r=result: self._on_tracal_result(r))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: (
                messagebox.showerror('tracal() failed', m),
                self._status_var.set('tracal() failed — see log'),
            ))
            logging.exception("tracal() failed")
        finally:
            self.after(0, self._check_processing_buttons)

    def _on_tracal_result(self, result: dict) -> None:
        """Main thread: store result, enable display RBs, switch to Transmittance view."""
        self._tracal_result = result
        n = len(result.get('header', {}).get('sample_labels', []))
        self._status_var.set(f'tracal() complete — {n} sample(s) saved')
        logging.info("tracal() complete — %d samples", n)
        self._check_processing_buttons()   # enables the Transmittance/Abs/OD RBs
        self._display_var.set('transmittance')
        self._redraw_plot()

    def _on_refcal(self) -> None:
        """Save reflect() results to disk in a background thread."""
        if self._save_fdir is None:
            return
        logging.info("refcal() started — folder: %s", self._save_fdir)
        self._btn_refcal.config(state='disabled')
        self._status_var.set('Running refcal()…')
        threading.Thread(
            target=self._run_refcal,
            args=(str(self._save_fdir),),
            daemon=True,
        ).start()

    def _run_refcal(self, fdir: str) -> None:
        """Background: call reflect(save=True) and post result to main thread."""
        try:
            result = refcal(fdir, save=True, ow=True)
            self.after(0, lambda r=result: self._on_refcal_result(r))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: (
                messagebox.showerror('refcal() failed', m),
                self._status_var.set('refcal() failed — see log'),
            ))
            logging.exception("refcal() failed")
        finally:
            self.after(0, self._check_processing_buttons)

    def _on_reflect_result(self, result: dict) -> None:
        """Main thread: store result, enable display RB, switch to Reflectance view."""
        self._refcal_result = result
        n = len(result.get('header', {}).get('sample_labels', []))
        self._status_var.set(f'refcal() complete — {n} sample(s) saved')
        logging.info("refcal() complete — %d samples", n)
        self._check_processing_buttons()   # enables the Reflectance RB
        self._display_var.set('reflectance')
        self._redraw_plot()

    # -----------------------------------------------------------------------
    # Window close
    # -----------------------------------------------------------------------

    def _on_close(self) -> None:
        """Confirm exit, then clean up matplotlib figures and destroy the window."""
        if not messagebox.askyesno(
            'Exit AutomateFTIR',
            'Are you sure you want to exit?\n\nMake sure all data has been saved.',
            icon='warning',
        ):
            return
        self._stop_live_poll()
        self._mm.disconnect()
        self._spec.disconnect()
        logging.getLogger().removeHandler(self._log_handler)
        logging.getLogger().removeHandler(self._file_log_handler)
        self._file_log_handler.close()
        plt.close('all')
        self.destroy()

    # -----------------------------------------------------------------------
    # Bench align
    # -----------------------------------------------------------------------

    def _on_bench_align(self) -> None:
        """Gate-check then send the bench-align DDE command on the main thread."""
        if not self._spec.connected:
            messagebox.showwarning(
                'Not connected', 'Spectrometer is not connected.')
            return
        if self._collection_active:
            messagebox.showwarning('Busy', 'A collection is already in progress.')
            return
        self._status_var.set('Starting bench alignment…')
        # Schedule the DDE exec to fire inside the messagebox event loop so the
        # dialog appears immediately rather than after the blocking Exec returns.
        self.after(100, self._fire_bench_align)
        messagebox.showinfo(
            'Bench Alignment Running',
            'OMNIC is performing the bench alignment.\n\n'
            'Wait for the alignment to finish, then close the OMNIC dialog.\n\n'
            'Click OK only after the OMNIC dialog is closed.')
        # The bench-align NACK leaves the DDE conversation in a broken state.
        # Disconnect and use the same poll loop as startup — OMNIC may need
        # several seconds to become responsive again after alignment.
        self._spec.disconnect()
        self._light_spectrometer.set_state('idle')
        self._spec_polling    = True
        self._spec_poll_start = time.monotonic()
        self._status_var.set('Reconnecting to OMNIC after alignment…')
        self.after(OMNIC_AUTOCONNECT_POLL_MS, self._spec_poll_connect)

    def _fire_bench_align(self) -> None:
        """Send the bench-align DDE command; NACK is expected and ignored."""
        try:
            self._spec.bench_align()
        except Exception as exc:
            logging.debug("Bench align NACK (expected): %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = AutomateFTIR()
    app.mainloop()


if __name__ == '__main__':
    main()
