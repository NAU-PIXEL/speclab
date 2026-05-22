#!/usr/bin/env python3
"""
speclab demo — emcal → SMA → plot

Steps
-----
0. (Optional) Build a custom endmember library interactively with SpeclibViewerLWIR,
   then export it.  Set USE_SPECLIBVIEWER = True below to activate this step.

1. Emission calibration (emcal).
   - Single-folder mode  (USE_MULTI_FOLDER = False): run on FDIR directly.
   - Multi-folder mode   (USE_MULTI_FOLDER = True):  scan FDIR for subfolders
     (or use FDIR_LIST for explicit paths), run emcal on each, then merge.

2. Spectral Mixture Analysis (SMA) against the chosen endmember library.

Usage
-----
    python demo.py
"""

import os
import tkinter as tk
from tkinter import filedialog

import numpy as np

import speclab
from speclab import emcal, sma, summary_sma, merge
from speclab import readDVhdf, printStructInfo, saveDVhdf, save_emcal_csv
from speclab.SpeclibViewerLWIR import SpeclibViewerLWIR

# =============================================================================
# USER CONFIGURATION
# =============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- Multi-folder options ----------------------------------------------------
# Set True to run emcal on every subfolder of example_data and merge results.
USE_MULTI_FOLDER = False

# Parent folder containing FTIR data.
# Single-folder mode  : must contain bbhot.CSV / bbwarm.CSV + sample CSVs.
# Multi-folder mode   : scanned for subfolders, each of which is a measurement set.
FDIR = (os.path.join(_HERE, 'example_data')
        if USE_MULTI_FOLDER else
        os.path.join(_HERE, 'example_data', 'WardRocks_igneous1'))

# Explicit list of subfolder paths.  Leave empty [] to auto-scan FDIR.
FDIR_LIST: list[str] = []

# Save merged result alongside the data after merging.
SAVE_MERGED = False

# --- Emcal options -----------------------------------------------------------
# Path to the measurement notes CSV.  None → search each folder automatically.
NOTES_PATH = None

# Blackbody temperatures (K).  None → read from notes file (lab='nau' only).
BB1 = None   # warm BB
BB2 = None   # hot  BB

# Emissivity retrieval method: 'nem', 'mmd', or 'hullfit'
METHOD = 'nem'
N_BB = 2

# Fitting wavenumber range
WN_RANGE_EMCAL = (400.0, 1600.0)

# --- Endmember library -------------------------------------------------------
# Set USE_SPECLIBVIEWER = True to launch SpeclibViewerLWIR and build a custom library.
# Set USE_SPECLIBVIEWER = False and supply ENDLIB_PATH to use an existing file.
USE_SPECLIBVIEWER = False

ENDLIB_PATH = os.path.join(_HERE, 'spectral_libraries', 'speclib_JFS_rock_forming_minerals.hdf')

# --- SMA options -------------------------------------------------------------
WN_RANGE_SMA = (300.0, 1700.0)
NOTCH_CO2      = 0      # wavenumber half-width to notch CO2 band; 0 to disable
SAVE_RESULTS   = False  # write SMA output to HDF5 + CSV alongside the data
SAVE_PLOTS     = False  # write individual plots as PNG files
SHOW_PLOTS     = True
PLOT_THRESHOLD = 5.0    # minimum concentration (%) to label endmembers in plot
GROUP          = True
PLOT_GROUP     = False   # display concentrations by mineral group in plot
PLOT_CUMULATIVE = True
PLOT_RESIDUAL  = True  # add residual panel (measured − modeled) to plot
PLOT_ERROR     = True  # show ± error values on endmember labels and pie chart
SLOPE          = True  # add synthetic slope endmember to SMA library
SAMPLE_T        = None  # (K) Imposed if not None for slope endmember

# =============================================================================
# STEP 1 — Emission calibration
# =============================================================================

if USE_MULTI_FOLDER:
    fdirs = FDIR_LIST if FDIR_LIST else sorted(
        f.path for f in os.scandir(FDIR) if f.is_dir()
    )
    print(f"\nRunning emcal on {len(fdirs)} subfolder(s) ...")

    results = []
    for fdir in fdirs:
        print(f"  {os.path.basename(fdir)} ...", end=' ', flush=True)
        r = emcal(
            fdir=fdir,
            bb1=BB1,
            bb2=BB2,
            n_bb=N_BB,
            lab='nau',
            notes_path=NOTES_PATH,
            method=METHOD,
            wn_range=WN_RANGE_EMCAL,
            save=SAVE_RESULTS,
            plot=False,
        )
        n = len(r.get('sample_labels', []))
        print(f"{n} sample(s)")
        results.append(r)

    print("  Merging ...")
    emcal_out = merge(*results, resample=True)
    n_total = len(emcal_out.get('sample_labels', []))
    print(f"  Merged result: {n_total} sample(s).")

    if SAVE_MERGED:
        fname = os.path.join(FDIR, 'emcal_merged_results.hdf')
        saveDVhdf(emcal_out, fname)
        save_emcal_csv(emcal_out, fname.replace('.hdf', '.csv'))
        print(f"  Saved → {fname}")

    if SHOW_PLOTS:
        from speclab.plot import plot_emcal
        plot_emcal(emcal_out, sort=True)

else:
    print("\nRunning emcal ...")
    emcal_out = emcal(
        fdir=FDIR,
        bb1=BB1,
        bb2=BB2,
        n_bb=N_BB,
        lab='nau',
        notes_path=NOTES_PATH,
        method=METHOD,
        wn_range=WN_RANGE_EMCAL,
        save=SAVE_RESULTS,
        plot=SHOW_PLOTS,
    )
    print(f"  emcal returned {len(emcal_out.get('sample_labels', []))} sample(s).")

# printStructInfo(emcal_out)

# =============================================================================
# STEP 2 (optional) — Build endmember library with SpeclibViewerLWIR
# =============================================================================

if USE_SPECLIBVIEWER:
    print("\nLaunching SpeclibViewerLWIR — build your album and use 'Export Album' to save.")
    app = SpeclibViewerLWIR()
    app.wait_window()

    exported = filedialog.askopenfilename(
        title="Select the exported endmember library (.hdf)",
        filetypes=[("HDF5 files", "*.hdf *.h5"), ("All files", "*")],
    )
    if not exported:
        raise SystemExit("No library selected — exiting.")
    ENDLIB_PATH = exported
    print(f"Using library: {ENDLIB_PATH}")

# =============================================================================
# STEP 3 — Spectral Mixture Analysis
# =============================================================================

print("\nRunning sma ...")

sma_out = sma(
    em_data=emcal_out,
    endlib=ENDLIB_PATH,
    wn_range=WN_RANGE_SMA,
    notchco2=NOTCH_CO2,
    bb=True,
    slope=SLOPE,
    sample_t=SAMPLE_T,
    group=GROUP,
    sort=True,
    calc_errors=True,
    plot=SHOW_PLOTS,
    plot_cumulative=PLOT_CUMULATIVE,
    save_plots=SAVE_PLOTS,
    plot_group=PLOT_GROUP,
    plot_residual=PLOT_RESIDUAL,
    plot_error=PLOT_ERROR,
    save=SAVE_RESULTS,
    save_path=FDIR,
)

summary_sma(sma_out, group=GROUP, threshold=PLOT_THRESHOLD)
