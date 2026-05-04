#!/usr/bin/env python3
"""
EmissionAutomation — GUI for automated emission spectroscopy data collection.

Layout
------
Top     : spectrometer control buttons + status display + progress bar
Main    : matplotlib spectral display (wavenumber bottom / wavelength top)
          | right panel with live multimeter readings
"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Allow running directly as a script in addition to normal entry points.
if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'speclab'

from .plot import _add_top_axis
from . import __version__

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
    'axes.grid':             True,
    'axes.axisbelow':        True,
    'axes.labelsize':        12,
    'axes.titlesize':        12,
    'grid.linestyle':        '--',
    'axes.formatter.limits': (-4, 4),
    'errorbar.capsize':      2,
})

# ---------------------------------------------------------------------------
# Channel metadata
# ---------------------------------------------------------------------------

_CHANNEL_LABELS: dict[int, str] = {
    101: '101: BB resistance low',
    102: '102: BB resistance high',
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

# Display order: resistance channels (high then low), separator, then temperature channels.
_PANEL_ORDER: list[int] = [102, 101, 103, 104, 105, 106, 107]

# Channel after which a visual separator is inserted in the multimeter panel.
_PANEL_SEPARATOR_AFTER: int = 101


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
# Main window
# ---------------------------------------------------------------------------

class EmissionAutomation(tk.Tk):
    """
    Top-level application window for automated emission data collection.

    Attributes
    ----------
    _ax : matplotlib.axes.Axes
        Primary spectral display axes (wavenumber, reversed).
    _secax : matplotlib.axes.Axes or None
        Secondary wavelength axis attached to the top of *_ax*.
    _save_fdir : Path or None
        Destination folder for saved spectra and metrics.  None until chosen.
    _progress_var : tk.DoubleVar
        Drives the progress bar (range 0–100).
    _status_var : tk.StringVar
        Short status message shown next to the progress bar.
    _plot_mode_var : tk.StringVar
        Spectral display mode: ``'stacked'`` or ``'single'``.
    _scale_mode_var : tk.StringVar
        Y-axis scaling: ``'common'`` (shared limits) or ``'normalized'``.
    _mm_mode_var : tk.StringVar
        Multimeter display mode: ``'live'`` or ``'last_sample'``.
    _channel_vars : dict[int, tk.StringVar]
        Live-updated display values for each multimeter channel.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title(f'EmissionAutomation v{__version__}')
        self.geometry('1400x860')
        self.minsize(1000, 620)

        # ── Processing state ─────────────────────────────────────────────────
        self._secax = None

        # ── Save directory ────────────────────────────────────────────────────
        self._save_fdir: Path | None  = None
        self._save_fdir_var           = tk.StringVar(value='(select folder to enable collection)')

        # ── Experiment type ───────────────────────────────────────────────────
        self._experiment_var = tk.StringVar(value='Emission MIR')

        # ── Control state ────────────────────────────────────────────────────
        self._progress_var = tk.DoubleVar(value=0.0)
        self._status_var   = tk.StringVar(value='Ready')

        # ── Plot controls ─────────────────────────────────────────────────────
        self._plot_mode_var  = tk.StringVar(value='stacked')
        self._scale_mode_var = tk.StringVar(value='common')

        # ── Multimeter display ────────────────────────────────────────────────
        self._mm_mode_var = tk.StringVar(value='live')
        self._channel_vars: dict[int, tk.StringVar] = {
            ch: tk.StringVar(value='—') for ch in _PANEL_ORDER
        }

        self._build_ui()
        self._refresh_buttons()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_top_frame()
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=(2, 0))
        self._build_main_frame()

    # ── Top frame: buttons + status/progress ────────────────────────────────

    def _build_top_frame(self) -> None:
        top = ttk.Frame(self, padding=(8, 6, 8, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        # Row 1 — spectrometer buttons (left) + save folder path (right)
        btn_row = ttk.Frame(top)
        btn_row.pack(side=tk.TOP, fill=tk.X)

        self._btn_collect_sample = ttk.Button(
            btn_row, text='Collect Sample', command=self._on_collect_sample)
        self._btn_collect_sample.pack(side=tk.LEFT, padx=(0, 4))

        self._btn_collect_bb = ttk.Button(
            btn_row, text='Collect BB', command=self._on_collect_bb)
        self._btn_collect_bb.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(btn_row, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10, pady=2)

        ttk.Label(btn_row, text='Experiment:').pack(side=tk.LEFT, padx=(0, 4))
        self._cb_experiment = ttk.Combobox(
            btn_row,
            textvariable=self._experiment_var,
            values=[
                'Emission MIR',
                'Emission MIR - Heated Samples',
                'Emission FIR',
                'Emission FIR - Heated Samples',
            ],
            state='readonly',
            width=28,
        )
        self._cb_experiment.pack(side=tk.LEFT, padx=(0, 8))

        self._btn_bench_align = ttk.Button(
            btn_row, text='Bench Align', command=self._on_bench_align)
        self._btn_bench_align.pack(side=tk.LEFT)

        # Folder path — right edge of the same row (pack RIGHT, rightmost first)
        ttk.Button(
            btn_row, text='Browse…', command=self._on_browse_folder,
        ).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Entry(
            btn_row,
            textvariable=self._save_fdir_var,
            state='readonly',
            width=48,
        ).pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Label(btn_row, text='Save to:').pack(side=tk.RIGHT, padx=(0, 4))
        ttk.Separator(btn_row, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=10, pady=2)

        # Row 2 — status label + progress bar
        info_row = ttk.Frame(top)
        info_row.pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        ttk.Label(info_row, text='Status:').pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(
            info_row,
            textvariable=self._status_var,
            foreground='gray',
            width=32,
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 12))

        # Connection indicators — right side of the info row
        ttk.Label(info_row, text='Multimeter:').pack(side=tk.RIGHT, padx=(6, 2))
        self._light_multimeter = IndicatorLight(info_row)
        self._light_multimeter.pack(side=tk.RIGHT)

        ttk.Separator(info_row, orient=tk.VERTICAL).pack(
            side=tk.RIGHT, fill=tk.Y, padx=8, pady=2)

        ttk.Label(info_row, text='Spectrometer:').pack(side=tk.RIGHT, padx=(6, 2))
        self._light_spectrometer = IndicatorLight(info_row)
        self._light_spectrometer.pack(side=tk.RIGHT)

        self._progress_bar = ttk.Progressbar(
            info_row,
            variable=self._progress_var,
            mode='determinate',
            maximum=100,
            length=300,
        )
        self._progress_bar.pack(side=tk.LEFT)

        self._pct_label = ttk.Label(info_row, text='0 %', width=6, anchor=tk.W)
        self._pct_label.pack(side=tk.LEFT, padx=(4, 0))

    # ── Main area: spectral plot (left) + multimeter panel (right) ───────────

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

        # ── Matplotlib canvas ─────────────────────────────────────────────
        self._fig = Figure(figsize=(10, 5), dpi=100)
        self._ax  = self._fig.add_subplot(111)
        self._fig.subplots_adjust(top=0.87, bottom=0.10, left=0.08, right=0.97)

        self._ax.set_xlabel('Wavenumber (cm⁻\xb9)')
        self._ax.set_ylabel('Single Beam')
        self._ax.set_xlim(2000, 200)

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self._canvas, plot_frame)

        self._secax = _add_top_axis(self._ax)
        self._canvas.draw()

    # ── Right panel: spectra listbox (top) + multimeter (bottom) ────────────

    def _build_right_panel(self, parent: ttk.PanedWindow) -> None:
        right = ttk.Frame(parent, width=290)
        right.pack_propagate(False)
        parent.add(right, weight=0)

        # ── Collected spectra listbox ──────────────────────────────────────
        spec_lf = ttk.LabelFrame(right, text='Collected Spectra', padding=4)
        spec_lf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=(2, 4))

        sb = ttk.Scrollbar(spec_lf, orient=tk.VERTICAL)
        self._spectra_lb = tk.Listbox(
            spec_lf,
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

        # ── Multimeter section ─────────────────────────────────────────────
        self._build_multimeter_panel(right)

    def _build_multimeter_panel(self, parent: tk.Widget) -> None:
        mm_frame = ttk.LabelFrame(parent, text='Multimeter', padding=(10, 6))
        mm_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(0, 2))

        # ── Mode controls ──────────────────────────────────────────────────
        ctrl_row = ttk.Frame(mm_frame)
        ctrl_row.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))

        ttk.Radiobutton(
            ctrl_row, text='Live', variable=self._mm_mode_var, value='live',
            command=self._on_mm_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(
            ctrl_row, text='Last Sample', variable=self._mm_mode_var,
            value='last_sample', command=self._on_mm_mode_change,
        ).pack(side=tk.LEFT, padx=(0, 10))
        self._btn_refresh = ttk.Button(
            ctrl_row, text='Refresh', command=self._on_refresh, width=9)
        self._btn_refresh.pack(side=tk.LEFT)

        ttk.Separator(mm_frame, orient=tk.HORIZONTAL).grid(
            row=1, column=0, columnspan=3, sticky=tk.EW, pady=(4, 6))

        grid_row = 2
        for ch in _PANEL_ORDER:
            ttk.Label(mm_frame, text=f'{_CHANNEL_LABELS[ch]}:', anchor=tk.W).grid(
                row=grid_row, column=0, sticky=tk.W, padx=(0, 10), pady=2)
            ttk.Label(
                mm_frame,
                textvariable=self._channel_vars[ch],
                anchor=tk.E,
                width=9,
                font=('TkFixedFont', 11),
            ).grid(row=grid_row, column=1, sticky=tk.E, padx=(0, 4), pady=2)
            ttk.Label(
                mm_frame,
                text=_CHANNEL_UNITS[ch],
                anchor=tk.W,
                foreground='gray',
            ).grid(row=grid_row, column=2, sticky=tk.W, pady=2)
            grid_row += 1

            if ch == _PANEL_SEPARATOR_AFTER:
                ttk.Separator(mm_frame, orient=tk.HORIZONTAL).grid(
                    row=grid_row, column=0, columnspan=3,
                    sticky=tk.EW, pady=(4, 4))
                grid_row += 1

    # -----------------------------------------------------------------------
    # Plot helpers
    # -----------------------------------------------------------------------

    def _refresh_plot(self) -> None:
        """Redraw the spectral canvas, rebuilding the secondary wavelength axis."""
        if self._secax is not None:
            try:
                self._secax.remove()
            except Exception:
                pass
            self._secax = None
        self._secax = _add_top_axis(self._ax)
        self._canvas.draw_idle()

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
            Measured value; pass None to reset the field to '—'.
        """
        if channel not in self._channel_vars:
            return
        self._channel_vars[channel].set('—' if value is None else f'{value:.2f}')

    def update_all_channels(self, readings: dict) -> None:
        """
        Bulk-update all multimeter channels.

        Parameters
        ----------
        readings : dict[int, float]
            Mapping of channel number → measured value.
        """
        for ch, val in readings.items():
            self.update_channel(ch, val)

    # -----------------------------------------------------------------------
    # Progress helpers
    # -----------------------------------------------------------------------

    def set_progress(self, value: float, status: str = '') -> None:
        """
        Update the progress bar and optional status message.

        Parameters
        ----------
        value : float
            Completion percentage, 0–100.
        status : str
            Short status string to display; unchanged if empty.
        """
        self._progress_var.set(value)
        self._pct_label.config(text=f'{value:.0f} %')
        if status:
            self._status_var.set(status)

    # -----------------------------------------------------------------------
    # Folder management
    # -----------------------------------------------------------------------

    def _on_browse_folder(self) -> None:
        """Open a directory chooser and update the save path."""
        chosen = filedialog.askdirectory(
            title='Select save folder',
            initialdir=str(self._save_fdir) if self._save_fdir else '.',
        )
        if not chosen:
            return
        self._save_fdir = Path(chosen)
        self._save_fdir_var.set(str(self._save_fdir))
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        """Enable or disable collection buttons based on whether a save folder is set."""
        state = 'normal' if self._save_fdir is not None else 'disabled'
        self._btn_collect_sample.config(state=state)
        self._btn_collect_bb.config(state=state)

    # -----------------------------------------------------------------------
    # Multimeter mode
    # -----------------------------------------------------------------------

    def _on_mm_mode_change(self) -> None:
        """Called when the Live / Last Sample radio selection changes."""
        pass

    def _on_refresh(
            self) -> None:
        """Trigger a one-shot multimeter reading."""
        pass

    # -----------------------------------------------------------------------
    # Spectra list
    # -----------------------------------------------------------------------

    def _on_spectra_select(self, _event=None) -> None:
        """Called when a spectrum entry is selected in the listbox."""
        pass

    # -----------------------------------------------------------------------
    # Button stubs — hardware control wired in later
    # -----------------------------------------------------------------------

    def _on_collect_sample(self) -> None:
        pass

    def _on_collect_bb(self) -> None:
        pass

    def _on_bench_align(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = EmissionAutomation()
    app.mainloop()


if __name__ == '__main__':
    main()