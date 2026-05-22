"""
speclab — spectroscopy processing library.

Provides emissivity calibration (NEM, hullfit), spectral utility functions,
water-vapour correction, spectral mixture analysis, and plotting tools.
"""

from .functions import (
    MissingTempsError,
    load_sbm,
    cal_rad,
    emcal,
    tracal,
    refcal,
    emissivity_nem,
    emissivity_alpha,
    emissivity_hullfit,
    emissivity_hullfit_linear,
    emissivity_mmd,
    dehyd,
    sma,
    summary_sma,
    merge,
    sort_cube,
    sum_group_conc,
    scan_sample_labels,
    resample_spectrum,
    insert_plot_gaps,
    save_instrument_grids,
    load_instrument_grids,
    INSTRUMENT_PRESETS,
)
from .utils import (
    bbt,
    rad,
    rad2wl,
    rad2wn,
    normalize,
    wn2wl,
    wl2wn,
    c2k,
    k2c,
    c2f,
    f2c,
    r2t_lo,
    r2t_hi,
    r2t_swri,
    r2t_nau,
    findFiles,
    readEmissionTXTnotes,
    readEmissionCSVnotes,
    readOMNIC,
    loadReflectanceCSV,
    saveReflectanceCSV,
    loadASD,
    readDVhdf,
    saveDVhdf,
    dv_to_album,
    save_emcal_csv,
    save_sma_csv,
    printStructInfo,
    recursiveHDFreader,
)
from . import plot
from .config import configure, get_config

__version__ = "0.2.0"
