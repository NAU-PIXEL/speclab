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
    read_tes,
    is_tes_result,
    sort_cube,
    sum_group_conc,
    scan_sample_labels,
    resample_spectrum,
    insert_plot_gaps,
    save_instrument_grids,
    load_instrument_grids,
    INSTRUMENT_PRESETS,
    moving_average,
    # single-spectrum VSWIR primitives
    remove_continuum,
    band_parameters,
    smooth_spectrum,
    detect_bands,
    # batch VSWIR processing
    smooth_spectra,
    remove_continuum_batch,
    band_parameters_batch,
    detect_bands_batch,
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
    load_reflectance_vswir,
    loadReflectanceCSV,
    saveReflectanceCSV,
    loadASD,
    readHDF,
    saveHDF,
    save_emcal_csv,
    save_sma_csv,
    printStructInfo,
    save_band_parameters_csv,
    load_band_parameters_csv,
)
from . import plot
from .config import configure, get_config

__version__ = "0.9.1"
