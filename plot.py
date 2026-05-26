#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting functions for emcal, tracal, sma, and VSWIR band-parameter outputs.

Called automatically from emcal(), tracal(), and sma() when plot=True.
Can also be called directly on saved output dicts.

Provides
--------
plot_emcal               Summary emissivity plot (+ optional radiance details panel).
plot_tracal              Two-panel transmittance summary plot.
plot_sma                 Per-sample spectral mixture analysis overlay plots.
plot_band_parameters     Scatter + ridge plots for band_parameters_batch() output.
plot_instrument_metrics  Temperature channels and BB resistance vs. time (dict or CSV path).
"""

import logging
import os

import numpy as np
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

# 40-color palette matching the viewer's analysis-tab _AN_COLORS.
# tab20 dark then light, then tab20b dark then light.
_PLOT_COLORS: list = (
    [plt.cm.tab20(i)  for i in range(0, 20, 2)]
    + [plt.cm.tab20(i)  for i in range(1, 20, 2)]
    + [plt.cm.tab20b(i) for i in range(0, 20, 2)]
    + [plt.cm.tab20b(i) for i in range(1, 20, 2)]
)
_PLOT_COLOR_OTHER = '#cccccc'

# Band-parameter metric registry: (key, display_label_template, is_wavelength_metric)
# Display labels use '{unit}' as a placeholder replaced at call time.
_BP_METRICS: list[tuple[str, str, bool]] = [
    ('wl_center',          'Band center ({unit})',  True),
    ('wl_min',             'Band min ({unit})',     True),
    ('band_depth',         'Depth',                 False),
    ('fwhm',               'FWHM ({unit})',         True),
    ('base_width',         'Base width ({unit})',   True),
    ('band_area',          'Band area ({unit})',    True),
    ('band_area_ratio',    'Area ratio',            False),
    ('asymmetry_hw',       'Asymmetry (HW)',        False),
    ('asymmetry_centroid', 'Asymmetry (centroid)',  False),
]
_BP_METRIC_KEYS:   list[str]      = [k         for k, _, _   in _BP_METRICS]
_BP_METRIC_LABELS: dict[str, str] = {k: lbl    for k, lbl, _ in _BP_METRICS}
_BP_METRIC_SCALED: set[str]       = {k         for k, _, sc  in _BP_METRICS if sc}


def _fw(x: float) -> float:
    """Convert wavenumber (cm⁻¹) to wavelength (µm)."""
    # matplotlib probes tick candidates including zero; nan is silently skipped.
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x != 0, 10000.0 / x, np.nan)


def _add_top_axis(ax: plt.Axes, which: str = "wl") -> plt.Axes | None:
    """
    Attach a secondary x-axis on top showing wavelength in µm or wavenumber
    in cm⁻¹.  Returns the secondary axes object, or None if the primary axis
    limits are not strictly positive (which would make 1e4/x undefined).
    The caller should remove the returned object before redrawing.
    """
    lo, hi = ax.get_xlim()
    if not (lo > 0 and hi > 0):
        return None

    secax = ax.secondary_xaxis('top', functions=(_fw, _fw))
    if which == "wn":
        # Primary is µm → top shows cm⁻¹
        secax.xaxis.set_ticks(
            [500, 800, 1000, 1500, 2000, 3000, 4000, 5000])
        secax.set_xlabel('Wavenumber (cm\u207b\xb9)')
    elif which == "wl":
        # Primary is cm⁻¹ → top shows µm
        wl_lo, wl_hi = sorted([1e4 / lo, 1e4 / hi])
        visible = [t for t in _WL_TICKS_UM if wl_lo <= t <= wl_hi]
        if len(visible) > _WL_TICKS_MAX:
            step = (len(visible) + _WL_TICKS_MAX - 1) // _WL_TICKS_MAX
            visible = visible[::step]
        if visible:
            secax.xaxis.set_ticks(visible)
        secax.set_xlabel('Wavelength (\u03bcm)')
    return secax


# Candidate wavelength tick positions (µm) spanning VNIR → FIR.
_WL_TICKS_UM: list = [
    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,   # VNIR
    1.25, 1.5, 2.0, 2.5,                  # SWIR
    3, 5, 7, 10, 12, 15, 20,              # MWIR / TIR  (strategic, not every integer)
    25, 30, 50, 75, 100, 150, 200,        # FIR
]
# Maximum ticks to show before thinning kicks in.
_WL_TICKS_MAX = 12


# =============================================================================
# plot_emcal
# =============================================================================

def plot_emcal(
    out: dict,
    plot_details: bool = False,
    max_spectra: int = 10,
    sort: bool = False,
    save_plots: bool = False,
) -> None:
    """
    Display emissivity summary and (optionally) radiance detail plots.

    When the number of spectra exceeds *max_spectra*, a navigable
    single-panel figure with Prev / Next buttons is shown instead of
    overloading one axes.

    Parameters
    ----------
    out : dict
        Output dict returned by :func:`~functions.emcal`.  Required keys:
        ``xaxis``, ``emiss``, ``label``.  For *plot_details*, also
        requires ``rad``, ``rad0``, ``sample_temps``.  The optional keys
        ``method`` and ``max_emiss`` are used for the figure title.
    plot_details : bool
        If True, show a radiance panel below the emissivity panel.
    max_spectra : int
        Maximum number of spectra per page.  Default 10.
    sort : bool
        If True, sort spectra alphabetically by label before plotting.
        Default False.
    save_plots : bool
        If True, batch-save all pages as PNG files to ``'./emcal_plots/'``
        before opening the interactive panel.
    """
    import matplotlib.widgets as mwidgets

    xaxis      = out['xaxis']
    labels     = sorted(out['label']) if sort else list(out['label'])
    n          = len(labels)
    method_str = out.get('method', '')
    max_emiss  = out.get('max_emiss', None)

    if method_str and max_emiss is not None:
        run_title = (
            f'EMISSION RUN: METHOD={method_str} | '
            f'MAX EMISSIVITY={max_emiss:.2f}'
        )
    else:
        run_title = 'EMISSION RUN'

    n_pages = max(1, (n + max_spectra - 1) // max_spectra)
    _default_save_dir = './emcal_plots'

    # ------------------------------------------------------------------
    # Content renderer — draws page p into one or two pre-cleared axes
    # ------------------------------------------------------------------
    def _draw(ax_em: plt.Axes, ax_rad: plt.Axes | None, p: int) -> None:
        start = p * max_spectra
        end   = min(start + max_spectra, n)
        page_labels = labels[start:end]

        for lbl in page_labels:
            ax_em.plot(xaxis, out['emiss'][lbl], label=lbl, zorder=0)

        ax_em.set_title(
            f'{run_title}  —  page {p + 1}/{n_pages}'
            f'  (spectra {start + 1}–{end} of {n})'
        )
        ax_em.set(xlabel='Wavenumber [cm⁻¹]', ylabel='Emissivity')
        ax_em.set_ylim(max(ax_em.get_ylim()[0], 0), min(ax_em.get_ylim()[1], 1.05))
        ax_em.invert_xaxis()
        ax_em.legend(fontsize=8)

        if ax_rad is not None:
            for lbl in page_labels:
                t_k = out['sample_temps'][lbl]
                ln, = ax_rad.plot(
                    xaxis, out['rad'][lbl],
                    label=f'{lbl} — {t_k:.1f} K',
                    zorder=0,
                )
                ax_rad.plot(
                    xaxis, out['rad0'][lbl],
                    ls='--', c=ln.get_color(), zorder=0,
                )
            ax_rad.set(xlabel='Wavenumber [cm⁻¹]', ylabel='Spectral Radiance')
            ax_rad.invert_xaxis()
            ax_rad.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Clean single-page save (no buttons in the output PNG)
    # ------------------------------------------------------------------
    def _save_page(p: int, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f'emcal_page{p + 1:03d}_of{n_pages}.png')
        if plot_details:
            fig_s, (ax_em_s, ax_rad_s) = plt.subplots(
                2, 1, figsize=(12, 10), tight_layout=True,
            )
            _draw(ax_em_s, ax_rad_s, p)
            _add_top_axis(ax_em_s)
        else:
            fig_s, ax_em_s = plt.subplots(figsize=(12, 8), tight_layout=True)
            _draw(ax_em_s, None, p)
            _add_top_axis(ax_em_s)
        fig_s.savefig(path, dpi=150)
        plt.close(fig_s)
        logging.info("Saved %s", path)

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------
    if save_plots:
        for p in range(n_pages):
            _save_page(p, _default_save_dir)

    # ------------------------------------------------------------------
    # Single-page shortcut — plain figure, no buttons
    # ------------------------------------------------------------------
    if n_pages == 1:
        if plot_details:
            fig, (ax_em, ax_rad) = plt.subplots(
                2, 1, figsize=(12, 10), tight_layout=True,
            )
            _draw(ax_em, ax_rad, 0)
            _add_top_axis(ax_em)
        else:
            fig, ax_em = plt.subplots(figsize=(12, 8), tight_layout=True)
            _draw(ax_em, None, 0)
            _add_top_axis(ax_em)
        plt.show()
        return

    # ------------------------------------------------------------------
    # Multi-page interactive panel
    # ------------------------------------------------------------------
    if plot_details:
        fig    = plt.figure(figsize=(12, 10))
        ax_em  = fig.add_axes([0.08, 0.48, 0.87, 0.43])
        ax_rad = fig.add_axes([0.08, 0.15, 0.87, 0.28])
    else:
        fig    = plt.figure(figsize=(12, 8))
        ax_em  = fig.add_axes([0.08, 0.15, 0.87, 0.75])
        ax_rad = None

    ax_prev    = fig.add_axes([0.06, 0.03, 0.17, 0.05])
    ax_save    = fig.add_axes([0.29, 0.03, 0.17, 0.05])
    ax_saveall = fig.add_axes([0.53, 0.03, 0.17, 0.05])
    ax_next    = fig.add_axes([0.77, 0.03, 0.17, 0.05])

    btn_prev    = mwidgets.Button(ax_prev,    '◀  Prev')
    btn_save    = mwidgets.Button(ax_save,    'Save')
    btn_saveall = mwidgets.Button(ax_saveall, 'Save All')
    btn_next    = mwidgets.Button(ax_next,    'Next  ▶')

    state: dict = {'page': 0, 'secax': None}

    def _refresh(p: int) -> None:
        state['page'] = p
        if state['secax'] is not None:
            try:
                state['secax'].remove()
            except Exception:
                pass
            state['secax'] = None
        ax_em.cla()
        if ax_rad is not None:
            ax_rad.cla()
        _draw(ax_em, ax_rad, p)
        state['secax'] = _add_top_axis(ax_em)
        fig.canvas.draw_idle()

    def on_prev(_event) -> None:
        if state['page'] > 0:
            _refresh(state['page'] - 1)

    def on_next(_event) -> None:
        if state['page'] < n_pages - 1:
            _refresh(state['page'] + 1)

    def on_save(_event) -> None:
        _save_page(state['page'], _default_save_dir)

    def on_saveall(_event) -> None:
        for p in range(n_pages):
            _save_page(p, _default_save_dir)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_save.on_clicked(on_save)
    btn_saveall.on_clicked(on_saveall)

    _refresh(0)
    plt.show()

# =============================================================================
# plot_speclib
# =============================================================================

def plot_speclib(
    out: dict,
    resample: str = 'original',
    max_spectra: int = 10,
    search_terms: str | None = None,
) -> None:
    """
    Display spectral library spectra in a navigable single-panel figure.

    Spectra are divided into pages of *max_spectra* each.  Prev / Next
    buttons step between pages.

    Parameters
    ----------
    out : dict
        Album dict ``{spec_id: {'data', 'xaxis', 'sample_name', ...}}``
        as returned by :func:`~utils.readDVhdf` / ``_load_hdf``, or a
        flat dict with shared ``'xaxis'``, ``'data'``, and ``'spec_id'``
        keys.
    resample : str
        Reserved for future resampling options.  Currently unused.
    max_spectra : int
        Number of spectra shown per page.  Default 10.
    search_terms : str or None
        Reserved for future filtering.  Currently unused.
    """
    import matplotlib.widgets as mwidgets

    # ------------------------------------------------------------------
    # Unpack: album format vs flat shared-xaxis format
    # ------------------------------------------------------------------
    if 'xaxis' not in out:
        # Album format — each entry has its own xaxis
        keys   = list(out.keys())
        xaxes  = np.array([out[k]['xaxis'] for k in keys], dtype=object)  # (n,) of xaxis arrays
        data   = np.vstack([out[k]['data']  for k in keys])
        labels = [str(out[k].get('sample_name', k)) for k in keys]
        shared_xaxis = None
    else:
        shared_xaxis = np.asarray(out['xaxis'], dtype=np.float64)
        data         = np.atleast_2d(np.asarray(out['data'], dtype=np.float64))
        labels       = [
            f"{sid}: {name}"
            for sid, name in zip(out['spec_id'], out['sample_name'])
        ]
        xaxes = None

    n       = len(labels)
    n_pages = max(1, (n + max_spectra - 1) // max_spectra)

    # ------------------------------------------------------------------
    # Content renderer — draws page p into any pre-cleared axes
    # ------------------------------------------------------------------
    def _draw(ax: plt.Axes, p: int) -> None:
        start = p * max_spectra
        end   = min(start + max_spectra, n)

        for i in range(start, end):
            x = shared_xaxis if shared_xaxis is not None else xaxes[i]
            ax.plot(x, data[i], label=labels[i], zorder=0)

        ax.set_title(f'Page {p + 1} of {n_pages}  —  spectra {start + 1}–{end} of {n}')
        ax.set_xlabel('Wavenumber [cm⁻¹]')
        ax.set_ylabel('Emissivity')
        ax.set_ylim(max(ax.get_ylim()[0], 0), min(ax.get_ylim()[1], 1.05))
        ax.invert_xaxis()
        ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    # Single-page shortcut — no navigation needed
    # ------------------------------------------------------------------
    if n_pages == 1:
        fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
        _draw(ax, 0)
        _add_top_axis(ax)
        plt.show()
        return

    # ------------------------------------------------------------------
    # Interactive panel
    # ------------------------------------------------------------------
    fig     = plt.figure(figsize=(12, 8))
    main_ax = fig.add_axes([0.08, 0.15, 0.87, 0.75])

    ax_prev = fig.add_axes([0.25, 0.03, 0.20, 0.05])
    ax_next = fig.add_axes([0.55, 0.03, 0.20, 0.05])

    btn_prev = mwidgets.Button(ax_prev, '◀  Prev')
    btn_next = mwidgets.Button(ax_next, 'Next  ▶')

    state: dict = {'page': 0, 'secax': None}

    def _refresh(p: int) -> None:
        state['page'] = p
        if state['secax'] is not None:
            try:
                state['secax'].remove()
            except Exception:
                pass
            state['secax'] = None
        main_ax.cla()
        _draw(main_ax, p)
        state['secax'] = _add_top_axis(main_ax)
        fig.canvas.draw_idle()

    def on_prev(_event) -> None:
        if state['page'] > 0:
            _refresh(state['page'] - 1)

    def on_next(_event) -> None:
        if state['page'] < n_pages - 1:
            _refresh(state['page'] + 1)

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)

    _refresh(0)
    plt.show()



# =============================================================================
# plot_tracal
# =============================================================================

def plot_tracal(out: dict) -> None:
    """
    Display a two-panel transmittance summary plot.

    Parameters
    ----------
    out : dict
        Output dict returned by :func:`~functions.tracal`.  Required keys:
        ``wn``, ``tra``, ``header``.
    """
    wn       = out['wn']
    labels   = out['header']['sample labels']
    n        = len(labels)
    blank_T  = out['tra']['blank']
    meanT    = out['tra']['mean']
    stdT     = out['tra']['std']
    data     = np.stack([out['tra'][lbl] for lbl in labels])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[1, 3])
    fig.suptitle('TRANSMISSION RUN')

    # Blank panel
    axes[0].plot(wn, blank_T, label='Blank T', c='k')
    axes[0].legend()
    axes[0].set(ylabel='Transmittance')
    axes[0].invert_xaxis()
    axes[0].label_outer()
    _add_top_axis(axes[0])

    # Sample panel
    for i in range(n):
        axes[1].plot(wn, data[i], label=labels[i], zorder=0)
    axes[1].plot(wn, meanT, label='Mean', c='k', lw=2.0)
    axes[1].fill_between(
        wn, meanT - stdT, meanT + stdT,
        label=r'1 $\sigma$', color='k', alpha=0.3, zorder=0,
    )
    axes[1].set(xlabel='Wavenumber [$cm^{-1}$]', ylabel='Transmittance')
    axes[1].invert_xaxis()
    axes[1].legend()

    fig.tight_layout()
    plt.show()


# =============================================================================
# plot_sma
# =============================================================================

def plot_sma(
    out: dict,
    threshold: float = 5.0,
    save_plots: bool = False,
    group: bool = False,
    residual: bool = False,
    error: bool = False,
    cumulative: bool = True,
    other: bool = False,
    offset: float = 0.0,
) -> None:
    """
    Display SMA results as a navigable multi-panel figure, or save all
    samples silently as PNG files.

    Parameters
    ----------
    out : dict
        Output dict from :func:`~functions.sma`.  Required keys:
        ``xaxis``, ``measured``, ``modeled``, ``normconc``, ``labels``,
        ``rms``, ``endlib``.  Optional: ``sample_labels``.
    threshold : float
        Minimum normalised concentration (%) for an endmember to appear
        in the overlay and pie chart.  Default 5.0.
    save_plots : bool
        If True, all samples are saved as PNG files to a default directory.
    group : bool
        If True and ``out['grouped']`` is present, display concentrations
        by mineral group rather than individual endmembers.
    residual : bool
        If True, add a bottom panel showing the spectral residual
        (measured − modeled).
    error : bool
        If True, append ``± X%`` error values to endmember overlay labels.
    cumulative : bool (default = True)
        If True, render the overlay as a cumulative stacked fill plot,
        showing how each endmember contributes to pulling the emissivity
        below 1.0.
    other : bool (default = False)
        If True, aggregate all endmembers below ``threshold`` into a single
        "Other" entry shown in the overlay.
    offset : float (default = 0.0)
        Vertical offset (emissivity units) between successive endmember
        overlay spectra in individual (non-cumulative) mode.  Default 0.0 leaves
        all spectra in their natural range.
    """
    import matplotlib.widgets as mwidgets
    from speclab.functions import resample_spectrum   # lazy — avoids circular ref

    # ------------------------------------------------------------------
    # Unpack output dict
    # ------------------------------------------------------------------
    xaxis         = out['xaxis']
    measured      = np.atleast_2d(out['measured'])
    modeled       = np.atleast_2d(out['modeled'])
    normconc      = np.atleast_2d(out['normconc'])
    labels        = out['labels']
    rms           = np.atleast_1d(out['rms'])
    endlib        = out.get('endlib') or {}
    n_samples     = measured.shape[0]
    bb_normconc     = np.atleast_1d(out.get('bb_normconc',     np.zeros(n_samples)))
    slope_normconc  = np.atleast_1d(out.get('slope_normconc',  np.zeros(n_samples)))
    delta_t_est     = np.atleast_1d(out.get('delta_t_estimated', np.full(n_samples, np.nan)))
    sample_labels  = out.get('sample_labels') or [
        f'Sample {i}' for i in range(n_samples)
    ]
    wn_range      = out.get('wn_range', (float(xaxis.min()), float(xaxis.max())))
    wn_lo, wn_hi  = wn_range

    # ------------------------------------------------------------------
    # Select display data: grouped or individual
    # ------------------------------------------------------------------
    gp = out.get('grouped') if group and out.get('grouped') else None
    if group and gp is None:
        logging.warning("plot_sma: group=True but no grouped data in output — falling back to individual display.")

    if gp is not None:
        disp_normconc  = np.atleast_2d(gp['grouped_normconc'])
        disp_labels    = list(gp['grouped_labels'])
        disp_normerror = (np.atleast_2d(gp['grouped_normerror'])
                          if 'grouped_normerror' in gp else None)
    else:
        disp_normconc  = normconc
        disp_labels    = labels
        disp_normerror = (np.atleast_2d(out['normerror'])
                          if 'normerror' in out else None)

    # ------------------------------------------------------------------
    # Pre-compute normalised endmember overlay spectra (shared across samples)
    # lib_unit[label] = spectrum normalised to [0, 1] over wn_mask.
    # In group mode: average of all endmembers in that category.
    # In individual mode: match by "{sample_name} {spec_id}" or category.
    # ------------------------------------------------------------------
    wn_mask:  np.ndarray = (xaxis >= 500) & (xaxis <= 1500)
    lib_unit: dict[str, np.ndarray] = {}
    lib_raw:  dict[str, np.ndarray] = {}

    def _norm_spec(spec: np.ndarray) -> np.ndarray:
        spec_range = spec[wn_mask]
        lo, hi     = float(spec_range.min()), float(spec_range.max())
        return (spec - lo) / (hi - lo + 1e-12)

    if endlib and gp is not None:
        group_specs: dict[str, list[np.ndarray]] = {lbl: [] for lbl in disp_labels}
        for sid in endlib:
            entry = endlib[sid]
            cat   = str(entry.get('category', ''))
            if cat in group_specs:
                group_specs[cat].append(
                    resample_spectrum(entry['xaxis'], entry['data'], xaxis)
                )
        for lbl, specs in group_specs.items():
            if specs:
                mean_s        = np.mean(specs, axis=0)
                lib_unit[lbl] = _norm_spec(mean_s)
                lib_raw[lbl]  = mean_s

    elif endlib:
        label_specs: dict[str, list[np.ndarray]] = {lbl: [] for lbl in disp_labels}
        for sid in endlib:
            entry       = endlib[sid]
            spec        = resample_spectrum(entry['xaxis'], entry['data'], xaxis)
            entry_label = (f"{entry.get('sample_name', 'Unknown')} "
                           f"{entry.get('spec_id', sid)}")
            category    = str(entry.get('category', ''))
            for lbl in disp_labels:
                if entry_label == lbl or category == lbl:
                    label_specs[lbl].append(spec)
                    break
        for lbl, specs in label_specs.items():
            if specs:
                mean_s        = np.mean(specs, axis=0)
                lib_unit[lbl] = _norm_spec(mean_s)
                lib_raw[lbl]  = mean_s

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _shade_excluded(ax: plt.Axes) -> None:
        xlo, xhi = float(xaxis.min()), float(xaxis.max())
        kw = dict(color='gray', alpha=0.15, zorder=0, lw=0)
        if xlo < wn_lo:
            ax.axvspan(xlo, wn_lo, **kw)
        if wn_hi < xhi:
            ax.axvspan(wn_hi, xhi, **kw)

    # ------------------------------------------------------------------
    # Content renderer
    # Layout possibilities:
    #   lib_unit + residual  →  3 panels: [overlay | spec | resid]
    #   lib_unit only        →  2 panels: [overlay | spec]
    #   residual only        →  2 panels: [spec | resid]
    #   neither              →  1 panel:  [spec]
    # ax_top = overlay (None when no lib_unit)
    # ax_bot = measured/modeled (always)
    # ax_res = residual (None when residual=False)
    # Returns the secondary (wavelength) axis added to the topmost panel.
    # ------------------------------------------------------------------
    def _draw(
        ax_top: plt.Axes | None,
        ax_bot: plt.Axes,
        ax_res: plt.Axes | None,
        i: int,
    ):
        meas_i  = measured[i]
        model_i = modeled[i]
        ydata   = meas_i[wn_mask] if wn_mask.any() else meas_i
        ymin    = float(np.nanmin(ydata)) * 0.95
        ymax    = float(np.nanmax(ydata)) * 1.05

        if out.get('has_slope'):
            _dt  = delta_t_est[i]
            _dt_str = f' (ΔT≈{_dt:.1f} K)' if np.isfinite(_dt) else ''
            sl_str = f'  Slope = {slope_normconc[i]:.1f}%{_dt_str}'
        else:
            sl_str = ''
        title = (f'{sample_labels[i]}  —  BB = {bb_normconc[i]:.1f}%'
                 f'{sl_str}  —  RMS = {rms[i]:.4f}')

        # Endmember / group overlay panel
        if ax_top is not None:
            ax_top.axhline(1.0, color='gray', lw=0.8, ls='--', zorder=0)
            conc_i = disp_normconc[i]
            err_i  = disp_normerror[i] if (error and disp_normerror is not None) else None
            order  = np.argsort(conc_i)[::-1]

            above: list[tuple[int, str]] = [
                (j, disp_labels[j]) for j in order
                if conc_i[j] >= threshold and disp_labels[j] in lib_raw
            ]

            # Optional Other composite from below-threshold endmembers
            other_item: tuple[float, np.ndarray] | None = None
            if other:
                below_frac = 0.0
                below_wsum = np.zeros_like(xaxis)
                for j in order:
                    lb = disp_labels[j]
                    c  = conc_i[j]
                    if 0.0 < c < threshold and lb in lib_raw:
                        f = c / 100.0
                        below_frac += f
                        below_wsum += lib_raw[lb] * f
                if below_frac > 0.0:
                    other_item = (below_frac, below_wsum)

            if cumulative:
                stack_top = np.ones_like(xaxis)
                for rank, (j, lb) in enumerate(above):
                    frac      = conc_i[j] / 100.0
                    depth     = frac * (1.0 - lib_raw[lb])
                    stack_bot = stack_top - depth
                    lbl_str   = (f'{lb}  {conc_i[j]:.1f} ± {err_i[j]:.1f}%'
                                 if err_i is not None else f'{lb}  {conc_i[j]:.1f}%')
                    color = _PLOT_COLORS[rank % len(_PLOT_COLORS)]
                    ax_top.fill_between(xaxis, stack_bot, stack_top,
                                        alpha=0.65, color=color, label=lbl_str)
                    ax_top.plot(xaxis, stack_bot, color=color, lw=0.5, alpha=0.9)
                    stack_top = stack_bot
                if other_item is not None:
                    tot_frac, wsum = other_item
                    avg_raw   = wsum / tot_frac
                    depth     = tot_frac * (1.0 - avg_raw)
                    stack_bot = stack_top - depth
                    ax_top.fill_between(xaxis, stack_bot, stack_top, alpha=0.65,
                                        color='gray', label=f'Other  {tot_frac*100:.1f}%')
                    ax_top.plot(xaxis, stack_bot, color='gray', lw=0.5, alpha=0.9)
                    stack_top = stack_bot
                _shade_excluded(ax_top)
                y_floor = float(np.nanmin(stack_top))
                ax_top.set_ylim(max(y_floor * 0.98, 0.0), 1.02)
                ax_top.set_title(title)
                ax_top.set_ylabel('Emissivity (stacked)')
                ax_top.tick_params(labelbottom=False)
                ax_top.legend(fontsize=8)
            else:
                n_shown = len(above) + (1 if other_item is not None else 0)
                for rank, (j, lb) in enumerate(above):
                    frac         = conc_i[j] / 100.0
                    spec_display = (1.0 - frac) + lib_unit[lb] * frac + offset * rank
                    lbl_str      = (f'{lb}  {conc_i[j]:.1f} ± {err_i[j]:.1f}%'
                                    if err_i is not None else f'{lb}  {conc_i[j]:.1f}%')
                    color = _PLOT_COLORS[rank % len(_PLOT_COLORS)]
                    ax_top.plot(xaxis, spec_display, lw=1.0, color=color, label=lbl_str)
                if other_item is not None:
                    tot_frac, wsum = other_item
                    avg_raw   = wsum / tot_frac
                    r_range   = avg_raw[wn_mask]
                    lo, hi    = float(r_range.min()), float(r_range.max())
                    avg_normd = (avg_raw - lo) / (hi - lo + 1e-12)
                    spec_disp = (1.0 - tot_frac) + avg_normd * tot_frac
                    spec_disp += offset * len(above)
                    ax_top.plot(xaxis, spec_disp, lw=1.0, ls='--', color='gray',
                                label=f'Other  {tot_frac*100:.1f}%')
                _shade_excluded(ax_top)
                y_top = 1.05 + offset * max(0, n_shown - 1)
                ax_top.set_ylim(0.0, y_top)
                ax_top.set_title(title)
                ax_top.set_ylabel('Scaled Emissivity')
                ax_top.tick_params(labelbottom=False)
                ax_top.legend(fontsize=8)

        # Measured / modeled panel
        ax_bot.axhline(1.0, color='gray', lw=0.8, ls='--', zorder=0)
        ax_bot.plot(xaxis, meas_i,  'dimgray',      lw=1.0, label='Measured')
        ax_bot.plot(xaxis, model_i, 'darkorange', lw=1.0, label='Modeled')
        _shade_excluded(ax_bot)
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            ax_bot.set_ylim(max(ymin, 0.0), min(ymax, 1.1))
        if ax_top is None:
            ax_bot.set_title(title)
        ax_bot.set_ylabel('Emissivity')
        ax_bot.legend(fontsize=8)
        if ax_res is None:
            ax_bot.set_xlabel('Wavenumber [cm⁻¹]')
        else:
            ax_bot.tick_params(labelbottom=False)

        # Residual panel
        if ax_res is not None:
            resid_i = meas_i - model_i
            ax_res.plot(xaxis, resid_i, color='#c03030', lw=1.0)
            ax_res.axhline(0, color='gray', lw=0.8, ls='--')
            _shade_excluded(ax_res)
            ax_res.set_ylabel('Residual')
            ax_res.set_xlabel('Wavenumber [cm⁻¹]')

        # x-axis: set on ax_bot (gridspec sharex propagates to others)
        ax_bot.set_xlim(xaxis.min(), xaxis.max())
        ax_bot.invert_xaxis()

        # Secondary wavelength axis on the topmost panel
        topmost = ax_top if ax_top is not None else ax_bot
        return _add_top_axis(topmost)

    # ------------------------------------------------------------------
    # Figure factory — returns (fig, ax_top | None, ax_bot, ax_res | None)
    # ------------------------------------------------------------------
    def _make_fig(for_save: bool = False) -> tuple[plt.Figure, plt.Axes | None, plt.Axes, plt.Axes | None]:
        bp = 0.08 if for_save else 0.16        # bottom padding for buttons
        kw = dict(left=0.08, right=0.95)
        if lib_unit and residual:
            fig_f = plt.figure(figsize=(10, 9))
            gs = fig_f.add_gridspec(3, 1, height_ratios=[2, 4, 1],
                                    hspace=0.03, top=0.90, bottom=bp, **kw)
            ax_t = fig_f.add_subplot(gs[0])
            ax_b = fig_f.add_subplot(gs[1], sharex=ax_t)
            ax_r = fig_f.add_subplot(gs[2], sharex=ax_t)
            return fig_f, ax_t, ax_b, ax_r
        elif lib_unit:
            fig_f = plt.figure(figsize=(10, 8))
            gs = fig_f.add_gridspec(2, 1, height_ratios=[3, 5],
                                    hspace=0.03, top=0.90, bottom=bp, **kw)
            ax_t = fig_f.add_subplot(gs[0])
            ax_b = fig_f.add_subplot(gs[1], sharex=ax_t)
            return fig_f, ax_t, ax_b, None
        elif residual:
            fig_f = plt.figure(figsize=(10, 7))
            gs = fig_f.add_gridspec(2, 1, height_ratios=[5, 2],
                                    hspace=0.06, top=0.93, bottom=bp, **kw)
            ax_b = fig_f.add_subplot(gs[0])
            ax_r = fig_f.add_subplot(gs[1], sharex=ax_b)
            return fig_f, None, ax_b, ax_r
        else:
            fig_f, ax_b = plt.subplots(figsize=(10, 6))
            fig_f.subplots_adjust(top=0.93, bottom=bp, **kw)
            return fig_f, None, ax_b, None

    # ------------------------------------------------------------------
    # Clean single-sample save (no buttons in the output PNG)
    # ------------------------------------------------------------------
    def _save_sample_plot(i: int, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        safe_lbl = (sample_labels[i]
                    .replace(' ', '_').replace('/', '-').replace('\\', '-'))
        path = os.path.join(out_dir, f'sma_{i:03d}_{safe_lbl}.png')
        fig_s, ax_t_s, ax_b_s, ax_r_s = _make_fig(for_save=True)
        _draw(ax_t_s, ax_b_s, ax_r_s, i)
        fig_s.savefig(path, dpi=150)
        plt.close(fig_s)
        logging.info("Saved %s", path)

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------
    _default_save_dir = './sma_plots'
    if save_plots:
        for i in range(n_samples):
            _save_sample_plot(i, _default_save_dir)

    # ------------------------------------------------------------------
    # Pie chart helper
    # ------------------------------------------------------------------
    def _show_pie(i: int) -> None:
        pie_nc     = disp_normconc[i]
        pie_lbls   = disp_labels
        pie_err    = disp_normerror[i] if (error and disp_normerror is not None) else None
        order      = np.argsort(pie_nc)[::-1]

        p_labels, p_values, p_colors = [], [], []
        color_idx = 0
        for j in order:
            if pie_nc[j] >= threshold:
                lbl_str = (f'{pie_lbls[j]}\n{pie_nc[j]:.1f} ± {pie_err[j]:.1f}%'
                           if pie_err is not None
                           else f'{pie_lbls[j]}\n{pie_nc[j]:.1f}%')
                p_labels.append(lbl_str)
                p_values.append(float(pie_nc[j]))
                p_colors.append(_PLOT_COLORS[color_idx % len(_PLOT_COLORS)])
                color_idx += 1

        remainder = 100.0 - sum(p_values)
        if remainder > 0.5:
            p_labels.append(f'Other\n{remainder:.1f}%')
            p_values.append(remainder)
            p_colors.append(_PLOT_COLOR_OTHER)

        if not p_values:
            return

        bb_pct   = float(bb_normconc[i])
        sl_pct   = float(slope_normconc[i])
        mode_str = '  [grouped]' if gp is not None else ''
        if out.get('has_slope'):
            _dt     = delta_t_est[i]
            _dt_str = f'  ΔT≈{_dt:.1f} K' if np.isfinite(_dt) else ''
            sl_pie_str = f'  |  Slope: {sl_pct:.1f}%{_dt_str}'
        else:
            sl_pie_str = ''

        existing = state.get('pie_fig')
        if existing is not None and plt.fignum_exists(existing.number):
            plt.close(existing)

        fig_p, ax_p = plt.subplots(figsize=(6, 5.5))
        fig_p.suptitle(
            f'{sample_labels[i]}  —  endmember composition{mode_str}\n'
            f'Blackbody: {bb_pct:.1f}%{sl_pie_str}',
            fontsize=9,
        )
        ax_p.pie(p_values, labels=p_labels, colors=p_colors,
                 startangle=90, counterclock=False,
                 wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                 textprops={'fontsize': 8})
        ax_p.axis('equal')
        ax_p.set_position([0.15, 0.08, 0.7, 0.78])
        fig_p.show()
        state['pie_fig'] = fig_p

    # ------------------------------------------------------------------
    # Interactive panel
    # ------------------------------------------------------------------
    fig, ax_top, ax_bot, ax_res = _make_fig(for_save=False)

    ax_prev    = fig.add_axes([0.05, 0.03, 0.14, 0.06])
    ax_save    = fig.add_axes([0.22, 0.03, 0.14, 0.06])
    ax_pie     = fig.add_axes([0.39, 0.03, 0.14, 0.06])
    ax_saveall = fig.add_axes([0.56, 0.03, 0.14, 0.06])
    ax_next    = fig.add_axes([0.73, 0.03, 0.14, 0.06])

    btn_prev    = mwidgets.Button(ax_prev,    '◀  Prev')
    btn_save    = mwidgets.Button(ax_save,    'Save')
    btn_pie     = mwidgets.Button(ax_pie,     'Pie')
    btn_saveall = mwidgets.Button(ax_saveall, 'Save All')
    btn_next    = mwidgets.Button(ax_next,    'Next  ▶')

    state: dict = {'idx': 0, 'secax': None, 'pie_fig': None}

    def _refresh(i: int) -> None:
        state['idx'] = i
        if state['secax'] is not None:
            try:
                state['secax'].remove()
            except Exception:
                pass
            state['secax'] = None
        for ax in (ax_top, ax_bot, ax_res):
            if ax is not None:
                ax.cla()
        state['secax'] = _draw(ax_top, ax_bot, ax_res, i)
        if state['pie_fig'] is not None and plt.fignum_exists(state['pie_fig'].number):
            _show_pie(i)
        fig.canvas.draw_idle()

    def on_prev(_event) -> None:
        if state['idx'] > 0:
            _refresh(state['idx'] - 1)

    def on_next(_event) -> None:
        if state['idx'] < n_samples - 1:
            _refresh(state['idx'] + 1)

    def on_save(_event) -> None:
        _save_sample_plot(state['idx'], _default_save_dir)

    def on_saveall(_event) -> None:
        for i in range(n_samples):
            _save_sample_plot(i, _default_save_dir)

    def on_pie(_event) -> None:
        _show_pie(state['idx'])

    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_save.on_clicked(on_save)
    btn_saveall.on_clicked(on_saveall)
    btn_pie.on_clicked(on_pie)

    _refresh(0)
    plt.show()


# =============================================================================
# plot_band_parameters
# =============================================================================

def plot_band_parameters(
    results:      'dict | str | os.PathLike',
    *,
    x_metric:     str              = 'band_depth',
    y_metric:     str              = 'wl_min',
    ridge_metric: str              = 'band_depth',
    kind:         str              = 'both',
    features:     list[str] | None = None,
    color_by:     str              = 'feature',
    unit:         str              = 'nm',
    show_points:  bool             = True,
    save_path:    'str | None'     = None,
    show:         bool             = True,
) -> 'plt.Figure | None':
    """
    Scatter and / or ridge plots of band-parameter results.

    Accepts the dict returned by :func:`~functions.band_parameters_batch` or a
    path to a CSV file previously exported by
    :func:`~utils.save_band_parameters_csv`.

    Parameters
    ----------
    results : dict or path-like
        Output dict from :func:`~functions.band_parameters_batch` with keys
        ``'features'``, ``'results'``, and optionally ``'sources'``.  When a
        file path is given, :func:`~utils.load_band_parameters_csv` is called
        to reconstruct the same structure.
    x_metric : str
        Metric plotted on the scatter X-axis (default ``'band_depth'``).
        Valid keys: ``'wl_center'``, ``'wl_min'``, ``'band_depth'``,
        ``'fwhm'``, ``'base_width'``, ``'band_area'``, ``'band_area_ratio'``,
        ``'asymmetry_hw'``, ``'asymmetry_centroid'``.
    y_metric : str
        Metric plotted on the scatter Y-axis (default ``'wl_min'``).
    ridge_metric : str
        Metric whose distribution is shown in the ridge plot
        (default ``'band_depth'``).
    kind : str
        Which panels to draw.  ``'scatter'``, ``'ridge'``, or ``'both'``
        (default).
    features : list[str] or None
        Restrict the plot to this subset of feature names.  ``None`` includes
        all features in *results*.
    color_by : str
        Colour scheme for the scatter plot.  One of ``'feature'`` (default,
        one colour per band name), ``'group'`` (one colour per feature group),
        ``'source'`` (Data vs Library), or ``'none'`` (uniform steelblue).
    unit : str
        Wavelength unit for axis labels — ``'nm'`` or ``'µm'``.  When
        ``'µm'``, all wavelength-valued metrics are divided by 1000.
    show_points : bool
        Overlay individual data points as a jitter strip on the ridge plot
        (default True).
    save_path : str or None
        Path to save the figure (PNG / PDF / SVG …).  When ``None`` (default)
        the figure is displayed interactively.
    """
    from pathlib import Path as _Path
    from matplotlib.lines import Line2D

    # ── Resolve input ─────────────────────────────────────────────────────────
    if isinstance(results, (str, _Path, os.PathLike)):
        try:
            from .utils import load_band_parameters_csv
        except ImportError:
            raise NotImplementedError(
                "CSV loading requires speclab.utils.load_band_parameters_csv "
                "(not yet implemented). Pass the dict from band_parameters_batch() directly."
            )
        out = load_band_parameters_csv(_Path(results))
    else:
        out = results

    feat_list: list[dict] = list(out['features'])
    res_dict:  dict       = out['results']
    sources:   dict       = out.get('sources', {})

    if features is not None:
        _keep = set(features)
        feat_list = [f for f in feat_list if f['name'] in _keep]

    if not feat_list:
        logging.warning("plot_band_parameters: no features to plot.")
        return

    # ── Validate metric keys ──────────────────────────────────────────────────
    for _k, _param in [(x_metric, 'x_metric'), (y_metric, 'y_metric'),
                       (ridge_metric, 'ridge_metric')]:
        if _k not in _BP_METRIC_KEYS:
            raise ValueError(
                f"{_param}='{_k}' is not a recognised metric. "
                f"Choose from: {_BP_METRIC_KEYS}"
            )

    # ── Build flat records ────────────────────────────────────────────────────
    scale = 1e-3 if unit == 'µm' else 1.0

    def _lbl(key: str) -> str:
        return _BP_METRIC_LABELS[key].replace('{unit}', unit)

    records: list[dict] = []
    for sp_name, feat_results in res_dict.items():
        for feat in feat_list:
            bp = feat_results.get(feat['name'])
            if bp is None:
                continue
            rec: dict = {
                'spectrum': sp_name,
                'feature':  feat['name'],
                'group':    feat.get('group', ''),
                'source':   sources.get(sp_name, ''),
            }
            for key in _BP_METRIC_KEYS:
                raw = bp.get(key, np.nan)
                if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                    rec[key] = np.nan
                else:
                    rec[key] = float(raw) * (scale if key in _BP_METRIC_SCALED else 1.0)
            records.append(rec)

    if not records:
        logging.warning("plot_band_parameters: all band-parameter results are None.")
        return

    # ── Scatter panel ─────────────────────────────────────────────────────────
    def _plot_scatter(ax: plt.Axes) -> None:
        valid = [r for r in records
                 if not (np.isnan(r.get(x_metric, np.nan))
                         or np.isnan(r.get(y_metric, np.nan)))]
        if not valid:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            return

        feat_names   = [f['name'] for f in feat_list]
        mixed_src    = len({r.get('source', '') for r in valid} - {''}) > 1
        _src_markers = {'Data': 'o', 'Library': '^', '': 'o'}

        def _scatter_pts(pts: list[dict], color) -> None:
            for src, mkr in _src_markers.items():
                sub = [r for r in pts if r.get('source', '') == src]
                if sub:
                    ax.scatter([r[x_metric] for r in sub],
                               [r[y_metric]  for r in sub],
                               marker=mkr, s=30, alpha=0.75, color=color)

        legend_handles: list = []
        n_cols = 1

        if color_by == 'feature':
            f_color = {f: _PLOT_COLORS[i % len(_PLOT_COLORS)]
                       for i, f in enumerate(feat_names)}
            for f in feat_names:
                pts = [r for r in valid if r['feature'] == f]
                if pts:
                    _scatter_pts(pts, f_color[f])
                    legend_handles.append(
                        Line2D([0], [0], marker='s', linestyle='none',
                               color=f_color[f], markersize=7, label=f)
                    )
            n_cols = max(1, len(legend_handles) // 12)

        elif color_by == 'group':
            groups  = sorted(set(r['group'] for r in valid))
            g_color = {g: _PLOT_COLORS[i % len(_PLOT_COLORS)]
                       for i, g in enumerate(groups)}
            for g in groups:
                pts = [r for r in valid if r['group'] == g]
                _scatter_pts(pts, g_color[g])
                legend_handles.append(
                    Line2D([0], [0], marker='s', linestyle='none',
                           color=g_color[g], markersize=7,
                           label=g if g else '(no group)')
                )

        elif color_by == 'source':
            src_color = {'Data': _PLOT_COLORS[0], 'Library': _PLOT_COLORS[2], '': 'steelblue'}
            srcs_present = sorted(set(r.get('source', '') for r in valid) - {''}) or ['']
            for src in srcs_present:
                pts = [r for r in valid if r.get('source', '') == src]
                ax.scatter([r[x_metric] for r in pts], [r[y_metric] for r in pts],
                           s=30, alpha=0.75, color=src_color.get(src, 'steelblue'),
                           label=src if src else 'Unknown')
                legend_handles.append(
                    Line2D([0], [0], marker='s', linestyle='none',
                           color=src_color.get(src, 'steelblue'),
                           markersize=7, label=src if src else 'Unknown')
                )

        else:  # 'none'
            _scatter_pts(valid, 'steelblue')

        # Source-shape legend when both Data and Library are present
        if mixed_src and color_by not in ('source', 'none'):
            shape_handles = [
                Line2D([0], [0], marker='o', linestyle='none', color='gray',
                       markersize=6, label='Data'),
                Line2D([0], [0], marker='^', linestyle='none', color='gray',
                       markersize=6, label='Library'),
            ]
            legend_handles = shape_handles + legend_handles

        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, markerscale=1.1,
                      loc='best', framealpha=0.7, ncols=n_cols)

        ax.set_xlabel(_lbl(x_metric),  fontsize=11)
        ax.set_ylabel(_lbl(y_metric),  fontsize=11)
        ax.set_title(f'{_lbl(x_metric)} vs {_lbl(y_metric)}', fontsize=11)

    # ── Ridge panel ───────────────────────────────────────────────────────────
    def _plot_ridge(ax: plt.Axes) -> None:
        from scipy.stats import gaussian_kde

        feat_names = [f['name'] for f in feat_list]
        n          = len(feat_names)
        colors     = [_PLOT_COLORS[i % len(_PLOT_COLORS)] for i in range(n)]

        band_spacing = 1.0
        kde_height   = 0.70
        cloud_height = 0.28

        # Collect per-feature value arrays (filter out nan)
        plot_d:  list[np.ndarray] = []
        for name in feat_names:
            vals = np.array([
                r[ridge_metric] for r in records
                if r['feature'] == name
                and not np.isnan(r.get(ridge_metric, np.nan))
            ])
            plot_d.append(vals)

        all_chunks = [v for v in plot_d if len(v)]
        if not all_chunks:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', color='gray')
            return

        all_vals = np.concatenate(all_chunks)
        x_margin = max((all_vals.max() - all_vals.min()) * 0.08, 1e-6)
        x_range  = np.linspace(all_vals.min() - x_margin,
                               all_vals.max() + x_margin, 400)
        rng = np.random.default_rng(seed=0)

        baselines: list[float] = []
        n_skipped = 0
        for i, (name, vals, color) in enumerate(zip(feat_names, plot_d, colors)):
            baseline = float((n - 1 - i) * band_spacing)
            baselines.append(baseline)
            ax.axhline(baseline, color=color, linewidth=0.6, alpha=0.35, zorder=1)

            if len(vals) < 2:
                n_skipped += 1
                continue

            kde = gaussian_kde(vals, bw_method='scott')
            ky  = kde(x_range)
            ky  = ky / ky.max() * kde_height
            ax.fill_between(x_range, baseline, baseline + ky,
                            color=color, alpha=0.45, zorder=2)
            ax.plot(x_range, baseline + ky,
                    color=color, linewidth=1.4, alpha=0.9, zorder=3)

            if show_points:
                jitter = rng.uniform(-cloud_height, 0.0, len(vals))
                ax.scatter(vals, baseline + jitter,
                           s=4, alpha=0.45, color=color,
                           linewidths=0, zorder=4)

        ax.set_yticks(baselines)
        ax.set_yticklabels(feat_names, fontsize=9)
        ax.tick_params(axis='y', length=0)
        ax.set_ylim(-cloud_height - 0.15,
                    (n - 1) * band_spacing + kde_height + 0.15)
        ax.set_xlabel(_lbl(ridge_metric), fontsize=11)
        ax.set_title(f'Distribution of {_lbl(ridge_metric)} by band', fontsize=11)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ('left', 'right', 'top'):
            ax.spines[spine].set_visible(False)

        if n_skipped:
            ax.text(0.99, 0.99, f'{n_skipped} band(s) skipped (< 2 spectra)',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=8, color='gray')

    # ── Compose figure ────────────────────────────────────────────────────────
    if kind == 'both':
        fig, (ax_sc, ax_rd) = plt.subplots(1, 2, figsize=(15, 6))
        _plot_scatter(ax_sc)
        _plot_ridge(ax_rd)
        fig.tight_layout(rect=[0, 0, 1, 1])
    elif kind == 'scatter':
        fig, ax_sc = plt.subplots(figsize=(8, 6))
        _plot_scatter(ax_sc)
        fig.tight_layout()
    elif kind == 'ridge':
        n_feats = len(feat_list)
        fig_h   = max(5.0, 0.55 * n_feats + 1.5)
        fig, ax_rd = plt.subplots(figsize=(8, fig_h))
        fig.subplots_adjust(left=0.28, right=0.97, top=0.93, bottom=0.08)
        _plot_ridge(ax_rd)
    else:
        raise ValueError(
            f"kind must be 'scatter', 'ridge', or 'both', got '{kind!r}'"
        )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info("Saved %s", save_path)
    if show:
        plt.show()
        return None
    return fig


# =============================================================================
# plot_instrument_metrics
# =============================================================================

# Channel numbers → human-readable labels (mirrors EmissionProcessor._CHANNEL_LABELS)
_CH_LABELS: dict[int, str] = {
    101: 'BB resistance low',
    102: 'BB resistance high',
    103: 'Mirror',
    104: 'Chamber exterior',
    105: 'Chamber interior',
    106: 'Chamber door',
    107: 'Detector',
}

_TEMP_CHANNELS_DEFAULT = (103, 104, 105, 106, 107)
_BB_CHANNELS_DEFAULT   = (101, 102)


def plot_instrument_metrics(
    notes: dict | str,
    labels: list | None = None,
    temp_channels: list[int] | None = None,
    bb_channels: list[int] | None = None,
    save_path: str | None = None,
) -> None:
    """
    Plot instrument temperature channels and BB resistance vs. measurement time.

    Temperature channels are drawn on the primary (left) y-axis; BB resistance
    channels on a twin secondary (right) y-axis.  Each sample acquisition is
    marked with a dashed vertical line and a rotated label.

    Parameters
    ----------
    notes : dict or str
        Either the notes dict stored in the emcal output (keys: ``'dtime'``,
        ``'channel_103'`` … ``'channel_107'``, ``'bb_dtime'``, ``'bb_ch101'``,
        ``'bb_ch102'``), or a path to a CSV / XLS / XLSX notes file (or a
        directory containing one).  When a filepath is given, the file is read
        with :func:`~utils.readEmissionCSVnotes` and all rows are plotted.
    labels : list or None
        Row labels for per-sample annotations.  Required when *notes* is a
        dict; derived from the ``sample_name`` column when *notes* is a
        filepath.
    temp_channels : list[int] or None
        Subset of temperature channel numbers to plot (103–107).
        Default None plots all five.
    bb_channels : list[int] or None
        Subset of BB resistance channel numbers to plot (101, 102).
        Default None plots both.
    save_path : str or None
        If given, save the figure to this path instead of displaying it.
    """
    import pandas as pd
    import matplotlib.dates as mdates

    # ── Input normalisation ───────────────────────────────────────────────────
    if isinstance(notes, str):
        from .utils import readEmissionCSVnotes
        from .functions import _BB_WARM_PATTERNS, _BB_HOT_PATTERNS

        df = readEmissionCSVnotes(notes)

        _bbc_pat = '|'.join(_BB_WARM_PATTERNS)
        _bbh_pat = '|'.join(_BB_HOT_PATTERNS)
        _any_bb  = f'{_bbc_pat}|{_bbh_pat}'
        is_bb    = df['sample_name'].str.contains(_any_bb, case=False, regex=True)
        df_samp  = df[~is_bb]
        labels   = df_samp['sample_name'].tolist()

        def _dt_list(df_sub) -> list:
            if 'dtime' not in df_sub.columns:
                return []
            return [str(v) if pd.notna(v) else '' for v in df_sub['dtime']]

        def _ch_list(df_sub, col: str) -> list:
            if col not in df_sub.columns:
                return []
            return [float(v) if pd.notna(v) else float('nan') for v in df_sub[col]]

        # BB resistance: warm then hot, matching emcal embed order (gives 2 points
        # connected by a dashed line on the secondary axis).
        _bb_dtimes, _bb_ch101, _bb_ch102 = [], [], []
        for _pat in (_bbc_pat, _bbh_pat):
            _row = df[df['sample_name'].str.contains(_pat, case=False, regex=True)]
            if len(_row) > 0:
                _r0 = _row.iloc[0]
                _bb_dtimes.append(str(_r0['dtime']) if pd.notna(_r0.get('dtime')) else '')
                _bb_ch101.append(float(_r0['channel_101']) if 'channel_101' in _r0 and pd.notna(_r0['channel_101']) else float('nan'))
                _bb_ch102.append(float(_r0['channel_102']) if 'channel_102' in _r0 and pd.notna(_r0['channel_102']) else float('nan'))

        notes = {
            'dtime':       _dt_list(df_samp),
            'channel_103': _ch_list(df_samp, 'channel_103'),
            'channel_104': _ch_list(df_samp, 'channel_104'),
            'channel_105': _ch_list(df_samp, 'channel_105'),
            'channel_106': _ch_list(df_samp, 'channel_106'),
            'channel_107': _ch_list(df_samp, 'channel_107'),
            'bb_dtime':    _bb_dtimes,
            'bb_ch101':    _bb_ch101,
            'bb_ch102':    _bb_ch102,
        }
    elif labels is None:
        raise ValueError("labels must be provided when notes is a dict")

    temp_chs = list(temp_channels) if temp_channels is not None else list(_TEMP_CHANNELS_DEFAULT)
    bb_chs   = list(bb_channels)   if bb_channels   is not None else list(_BB_CHANNELS_DEFAULT)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def _parse_sort(dtime_raw) -> tuple[list | None, np.ndarray | None]:
        """Return (sorted valid-Timestamp list, index array) or (None, None)."""
        try:
            parsed = pd.to_datetime(list(dtime_raw), errors='coerce', format='mixed')
            valid  = np.array([pd.notna(t) for t in parsed])
            if valid.any():
                valid_idx   = np.where(valid)[0]
                sort_within = np.argsort(
                    [parsed[i].value for i in valid_idx], kind='stable'
                )
                final_idx = valid_idx[sort_within]
                return [parsed[i] for i in final_idx], final_idx
        except Exception:
            pass
        return None, None

    # ── Primary axis: temperature channels ───────────────────────────────────
    dtimes, sort_idx = _parse_sort(notes.get('dtime', []))
    sorted_labels = ([labels[i] for i in sort_idx]
                     if sort_idx is not None else list(labels))
    x_sample = dtimes if dtimes is not None else list(range(len(labels)))

    any_temp = False
    for c_idx, ch in enumerate(temp_chs):
        col      = f'channel_{ch}'
        name     = _CH_LABELS.get(ch, col)
        vals_raw = np.array(list(notes.get(col, [])), dtype=float)
        if not np.isfinite(vals_raw).any():
            continue
        vals = vals_raw[sort_idx] if sort_idx is not None else vals_raw
        ax.plot(x_sample, vals, label=name,
                color=colors[c_idx % len(colors)], marker='o', ms=4, lw=1.2)
        any_temp = True

    # Per-sample annotations: shaded window (±1 min), vertical line, label
    if dtimes is not None and any_temp:
        xform    = ax.get_xaxis_transform()
        half_dur = pd.Timedelta(minutes=1)
        for lbl, dt in zip(sorted_labels, dtimes):
            if pd.isna(dt):
                continue
            ax.axvspan(dt - half_dur, dt + half_dur,
                       color='gray', alpha=0.10, lw=0, zorder=0)
            ax.axvline(dt, color='gray', lw=0.7, ls='--', zorder=1)
            ax.text(dt, 1.0, lbl, rotation=90, va='top', ha='right',
                    fontsize=7, color='gray', transform=xform)

    # ── Secondary axis: BB resistance channels ────────────────────────────────
    bb_dtimes, bb_sort_idx = _parse_sort(notes.get('bb_dtime', []))
    bb_x = bb_dtimes if bb_dtimes is not None else list(range(
        len(notes.get('bb_dtime', []))))

    any_resist = False
    bb_color_offset = len(_TEMP_CHANNELS_DEFAULT)
    for b_idx, ch in enumerate(bb_chs):
        key  = f'bb_ch{ch}'
        name = _CH_LABELS.get(ch, key)
        raw  = np.array(list(notes.get(key, [])), dtype=float)
        if not np.isfinite(raw).any():
            continue
        vals  = raw[bb_sort_idx] if bb_sort_idx is not None else raw
        color = colors[(bb_color_offset + b_idx) % len(colors)]
        ax2.plot(bb_x, vals, label=name,
                 color=color, marker='s', ms=5, lw=1.2, ls='--')
        any_resist = True

    # ── Labels, legend, formatting ────────────────────────────────────────────
    use_dates = dtimes is not None or bb_dtimes is not None
    if use_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=30)

    ax.set_xlabel('Time')
    ax.set_ylabel('Temperature (°C)')
    if any_resist:
        ax2.set_ylabel('Resistance (Ω)')
    ax2.tick_params(axis='y', which='both',
                    right=any_resist, labelright=any_resist)

    lines1, lbls1 = ax.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax.legend(lines1 + lines2, lbls1 + lbls2, fontsize=8, loc='upper right')

    ax.grid(True, lw=0.4, alpha=0.5)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        logging.info("Saved %s", save_path)
        plt.close(fig)
    else:
        plt.show()