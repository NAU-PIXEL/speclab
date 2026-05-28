#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for spectroscopy data processing.

Provides
--------
SilentCall              Context manager to suppress stdout/stderr.
findFiles               Case-insensitive recursive file search.
c2k / k2c / c2f / f2c  Temperature unit conversions.
wn2wl / wl2wn           Wavenumber ↔ wavelength conversions.
normalize               Min-max normalisation.
rad                     Planck blackbody radiance.
bbt                     Brightness temperature from radiance.
rad2wl / rad2wn         Spectral radiance unit conversions.
r2t_lo / r2t_hi         PRT resistance → temperature (NAU PRTs 31985/31986).
r2t_swri                PRT resistance → temperature (SwRI PRTs).
r2t_nau                 PRT resistance → temperature (NAU standard, returns K).
readEmissionTXTnotes    Parse legacy TXT emission measurement notes.
readEmissionCSVnotes    Load CSV/XLS emission measurement notes.
readOMNIC               Read a two-column OMNIC CSV spectrum file.
printStructInfo         Recursively print a nested dict structure.
find_key_recursively    Yield all values for a key in a nested dict.
saveHDF                 Save a nested dict to an HDF5 file.
readHDF                 High-level wrapper: load + normalise an HDF5 file (DaVinci-compatible).
save_band_parameters_csv / load_band_parameters_csv
                        Round-trippable CSV I/O for band_parameters_batch() output.
"""
from collections.abc import Iterator

import logging
import pandas as pd
import numpy as np
import re
import fnmatch
import os
import sys
import h5py
from datetime import time


class SilentCall:
    """
    Context manager that redirects stdout and stderr to /dev/null (or a
    caller-supplied stream), suppressing terminal output from third-party code.

    Parameters
    ----------
    stdout : file-like or None
        Replacement for sys.stdout inside the block.  Defaults to /dev/null.
    stderr : file-like or None
        Replacement for sys.stderr inside the block.  Defaults to /dev/null.

    Examples
    --------
    >>> with SilentCall():
    ...     noisy_third_party_function()
    """

    def __init__(self, stdout=None, stderr=None):
        self.devnull = open(os.devnull, 'w')
        self._stdout = stdout if stdout is not None else self.devnull
        self._stderr = stderr if stderr is not None else self.devnull

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        
        

# =============================================================================
# ============================= findFiles =====================================
# =============================================================================
def findFiles(
    terms: str | list[str],
    extensions: str | list[str] = '.*',
    fdir: str = '.',
) -> list[str]:
    """
    Recursively find files whose names match one or more case-insensitive
    glob patterns.

    Parameters
    ----------
    terms : str or list of str
        Substring(s) to search for.  Each term is wrapped in ``*...*`` to
        produce a glob pattern.  Pass ``""`` to match all filenames.
    extensions : str or list of str
        File extension(s) to include (e.g. ``".csv"``).  Default ``".*"``
        matches any extension.
    fdir : str
        Root directory to search recursively.  Default is the current
        working directory.

    Returns
    -------
    list of str
        Absolute paths of all matching files, in discovery order.  A path
        is included at most once even if it matches multiple patterns.
    """
    if not isinstance(terms, list):
        terms = [terms]
    if not isinstance(extensions, list):
        extensions = [extensions]

    search_strings = ['*%s*' % term + ext for term in terms for ext in extensions]

    matching_files: list[str] = []
    for pattern in search_strings:
        regex = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
        for root, _, files in os.walk(fdir):
            for file in files:
                path = os.path.join(root, file)
                if regex.match(file) and path not in matching_files:
                    matching_files.append(path)

    return matching_files


# =============================================================================
# ========================== Unit Conversion ==================================
# =============================================================================
def c2k(temp: float | np.ndarray) -> float | np.ndarray:
    """Convert Celsius to Kelvin."""
    return temp + 273.15


def k2c(temp: float | np.ndarray) -> float | np.ndarray:
    """Convert Kelvin to Celsius."""
    return temp - 273.15


def c2f(temp: float | np.ndarray) -> float | np.ndarray:
    """Convert Celsius to Fahrenheit."""
    return temp * 9 / 5 + 32


def f2c(temp: float | np.ndarray) -> float | np.ndarray:
    """Convert Fahrenheit to Celsius."""
    return (temp - 32) * 5 / 9


def wn2wl(
    wn: float | np.ndarray,
    units: str = 'um',
) -> float | np.ndarray:
    """
    Convert wavenumber (cm⁻¹) to wavelength.

    Parameters
    ----------
    wn : float or np.ndarray
        Wavenumber(s) in cm⁻¹.
    units : str
        Output wavelength units: ``'um'`` (default), ``'nm'``, or ``'m'``.

    Returns
    -------
    float or np.ndarray
        Wavelength in the requested units.
    """
    wl = 1e4 / wn
    if units == 'nm':
        return wl * 1e3
    elif units == 'm':
        return wl * 1e-6
    return wl


def wl2wn(
    wl: float | np.ndarray,
    units: str = 'um',
) -> float | np.ndarray:
    """
    Convert wavelength to wavenumber (cm⁻¹).

    Parameters
    ----------
    wl : float or np.ndarray
        Wavelength value(s).
    units : str
        Input wavelength units: ``'um'`` (default), ``'nm'``, or ``'m'``.

    Returns
    -------
    float or np.ndarray
        Wavenumber(s) in cm⁻¹.
    """
    if units == 'nm':
        wl = wl / 1e3
    elif units == 'm':
        wl = wl * 1e6
    return 1e4 / wl


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalisation to an array.

    Parameters
    ----------
    data : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array scaled to [0, 1].
    """
    data_range = data.max() - data.min()
    return (data - data.min()) / data_range

# =============================================================================
# =============================== rad =========================================
# =============================================================================
def rad(
    wn: float | np.ndarray,
    temp: float,
    wl: bool = False,
    norm: bool = False,
) -> float | np.ndarray:
    """
    Compute Planck blackbody spectral radiance.

    Parameters
    ----------
    wn : float or np.ndarray
        Spectral axis.  Wavenumber in cm⁻¹ when *wl* is False; wavelength
        in µm when *wl* is True.
    temp : float
        Blackbody temperature (K).
    wl : bool
        If True, treat *wn* as wavelength in µm and return radiance in
        W m⁻² sr⁻¹ µm⁻¹.  If False (default), treat *wn* as wavenumber in
        cm⁻¹ and return radiance in mW m⁻² sr⁻¹ cm.
    norm : bool
        If True, normalise the output to its maximum value.

    Returns
    -------
    float or np.ndarray
        Planck radiance in W m⁻² sr⁻¹ µm⁻¹ (*wl* = True) or
        mW m⁻² sr⁻¹ cm (*wl* = False).
    """
    if wl:
        c1 = 1.191042e8   # W m⁻² sr⁻¹ µm⁴
        c2 = 1.4387752e4  # K µm
        L = c1 / (wn ** 5 * (np.exp(c2 / (wn * temp)) - 1))
    else:
        c1 = 1.191042e-5  # mW m⁻² sr⁻¹ cm⁴
        c2 = 1.4387752    # K cm
        L = c1 * wn ** 3 / (np.exp(c2 * wn / temp) - 1)

    if norm:
        return L / L.max()
    return L


# =============================================================================
# ================================= bbt =======================================
# =============================================================================
def bbt(
    wn: float | np.ndarray,
    rad: float | np.ndarray,
    wl: bool = False,
) -> float | np.ndarray:
    """
    Compute brightness temperature from spectral radiance (inverse Planck).

    Parameters
    ----------
    wn : float or np.ndarray
        Spectral axis.  Wavenumber in cm⁻¹ when *wl* is False; wavelength
        in µm when *wl* is True.
    rad : float or np.ndarray
        Spectral radiance in mW m⁻² sr⁻¹ cm.
    wl : bool
        If True, convert *wn* from µm to cm⁻¹ before computing.

    Returns
    -------
    float or np.ndarray
        Brightness temperature (K) at each spectral point.
    """
    if wl:
        wn = 1e4 / wn

    c1 = 1.191042e-5  # mW m⁻² sr⁻¹ cm⁴
    c2 = 1.4387752    # K cm

    return c2 * wn / np.log(c1 * wn ** 3 / rad + 1)


# =============================================================================
# =============================== rad2wl ======================================
# =============================================================================
def rad2wl(
    wn: float | np.ndarray,
    L1: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert spectral radiance from wavenumber to wavelength space.

    Parameters
    ----------
    wn : float or np.ndarray
        Wavenumber axis (cm⁻¹).
    L1 : float or np.ndarray
        Spectral radiance in mW m⁻² sr⁻¹ cm.

    Returns
    -------
    wl : float or np.ndarray
        Wavelength axis (µm).
    L2 : float or np.ndarray
        Spectral radiance in mW m⁻² sr⁻¹ µm⁻¹.
    """
    wl = 1e4 / wn
    L2 = L1 * wn ** 2 / 1e4
    return wl, L2


# =============================================================================
# =============================== rad2wn ======================================
# =============================================================================
def rad2wn(
    wl: float | np.ndarray,
    L1: float | np.ndarray,
    units: str = 'um',
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """
    Convert spectral radiance from wavelength to wavenumber space.

    Parameters
    ----------
    wl : float or np.ndarray
        Wavelength axis.
    L1 : float or np.ndarray
        Spectral radiance in mW m⁻² sr⁻¹ µm⁻¹.
    units : str
        Units of *wl*: ``'um'`` (default), ``'nm'``, or ``'m'``.

    Returns
    -------
    wn : float or np.ndarray
        Wavenumber axis (cm⁻¹).
    L2 : float or np.ndarray
        Spectral radiance in mW m⁻² sr⁻¹ cm.
    """
    if units == 'nm':
        wl = wl / 1e3
    elif units == 'm':
        wl = wl * 1e6

    wn = 1e4 / wl
    L2 = L1 * 1e4 ** 2 / wl
    return wn, L2

# =============================================================================
# =============================== r2t_lo ======================================
# =============================================================================
def r2t_lo(ch1: float, ch2: float) -> float:
    """
    Convert PRT resistance to temperature for the NAU warm blackbody.

    Uses the Callendar-Van Dusen equation with coefficients for NAU PRTs
    31985 (channel 1) and 31986 (channel 2), Rosemount manual v1.5.
    Returns the average of the two channel temperatures in °C.

    Parameters
    ----------
    ch1 : float
        Resistance reading from PRT channel 1 (Ω).
    ch2 : float
        Resistance reading from PRT channel 2 (Ω).

    Returns
    -------
    float
        Average blackbody temperature (°C).
    """
    A1, B1, rho1 = 3.946495e-3, -5.87549e-7, 1000.71666
    A2, B2, rho2 = 3.947915e-3, -5.8755e-7,  1001.59058

    W1  = ch1 / rho1
    tp1 = ((A1 ** 2 - 4 * B1 * (1 - W1)) ** 0.5 - A1) / (2 * B1)
    t1  = tp1 + 0.045 * (tp1 / 100) * (tp1 / 100 - 1) * (tp1 / 419.58 - 1) * (tp1 / 630.74 - 1)

    W2  = ch2 / rho2
    tp2 = ((A2 ** 2 - 4 * B2 * (1 - W2)) ** 0.5 - A2) / (2 * B2)
    t2  = tp2 + 0.045 * (tp2 / 100) * (tp2 / 100 - 1) * (tp2 / 419.58 - 1) * (tp2 / 630.74 - 1)

    return (t1 + t2) / 2

# =============================================================================
# =============================== r2t_hi ======================================
# =============================================================================
def r2t_hi(ch1: float, ch2: float) -> float:
    """
    Convert PRT resistance to temperature for the NAU hot blackbody.

    Uses the Callendar-Van Dusen equation with coefficients for NAU PRTs
    channels 3 and 4, Rosemount manual v1.5.
    Returns the average of the two channel temperatures in °C.

    Parameters
    ----------
    ch1 : float
        Resistance reading from PRT channel 3 (Ω).
    ch2 : float
        Resistance reading from PRT channel 4 (Ω).

    Returns
    -------
    float
        Average blackbody temperature (°C).
    """
    A3, B3, rho3 = 3.946065e-3, -5.8755e-7, 1001.01647
    A4, B4, rho4 = 3.947985e-3, -5.8755e-7, 1000.91666

    W3  = ch1 / rho3
    tp3 = ((A3 ** 2 - 4 * B3 * (1 - W3)) ** 0.5 - A3) / (2 * B3)
    t3  = tp3 + 0.045 * (tp3 / 100) * (tp3 / 100 - 1) * (tp3 / 419.58 - 1) * (tp3 / 630.74 - 1)

    W4  = ch2 / rho4
    tp4 = ((A4 ** 2 - 4 * B4 * (1 - W4)) ** 0.5 - A4) / (2 * B4)
    t4  = tp4 + 0.045 * (tp4 / 100) * (tp4 / 100 - 1) * (tp4 / 419.58 - 1) * (tp4 / 630.74 - 1)

    return (t3 + t4) / 2


# =============================================================================
# =============================== r2t_swri ====================================
# =============================================================================
def r2t_swri(ch1: float, ch2: float) -> float:
    """
    Convert PRT resistance to temperature for SwRI blackbodies.

    Uses the Callendar-Van Dusen equation with SwRI-specific PRT
    coefficients.  Returns the average of the two channel temperatures in °C.

    Parameters
    ----------
    ch1 : float
        Resistance reading from PRT channel 1 (Ω).
    ch2 : float
        Resistance reading from PRT channel 2 (Ω).

    Returns
    -------
    float
        Average blackbody temperature (°C).
    """
    A1, B1, rho1 = 3.947075e-3, -5.8755e-7, 1001.65018
    A2, B2, rho2 = 3.946875e-3, -5.8755e-7, 1000.40883

    W1  = ch1 / rho1
    tp1 = ((A1 ** 2 - 4 * B1 * (1 - W1)) ** 0.5 - A1) / (2 * B1)
    t1  = tp1 + 0.045 * (tp1 / 100) * (tp1 / 100 - 1) * (tp1 / 419.58 - 1) * (tp1 / 630.74 - 1)

    W2  = ch2 / rho2
    tp2 = ((A2 ** 2 - 4 * B2 * (1 - W2)) ** 0.5 - A2) / (2 * B2)
    t2  = tp2 + 0.045 * (tp2 / 100) * (tp2 / 100 - 1) * (tp2 / 419.58 - 1) * (tp2 / 630.74 - 1)

    return (t1 + t2) / 2


# =============================================================================
# =============================== r2t_nau =====================================
# =============================================================================
def r2t_nau(ch1: float, ch2: float) -> float:
    """
    Convert PRT resistance to temperature for NAU standard blackbodies.

    Uses the Callendar-Van Dusen equation with standard Pt100 coefficients
    (ITS-90 / IEC 60751).  Returns temperature in **Kelvin** (unlike
    ``r2t_lo`` / ``r2t_hi`` / ``r2t_swri`` which return °C).

    Parameters
    ----------
    ch1 : float
        Resistance reading from PRT channel 1 (Ω).
    ch2 : float
        Resistance reading from PRT channel 2 (Ω).

    Returns
    -------
    float
        Average blackbody temperature (K).
    """
    A1 = A2 = 3.9083e-3
    B1 = B2 = -5.775e-7
    rho1 = rho2 = 1000.0

    W1  = ch1 / rho1
    tp1 = ((A1 ** 2 - 4 * B1 * (1 - W1)) ** 0.5 - A1) / (2 * B1)
    t1  = tp1 + 0.045 * (tp1 / 100) * (tp1 / 100 - 1) * (tp1 / 419.58 - 1) * (tp1 / 630.74 - 1)

    W2  = ch2 / rho2
    tp2 = ((A2 ** 2 - 4 * B2 * (1 - W2)) ** 0.5 - A2) / (2 * B2)
    t2  = tp2 + 0.045 * (tp2 / 100) * (tp2 / 100 - 1) * (tp2 / 419.58 - 1) * (tp2 / 630.74 - 1)

    return c2k((t1 + t2) / 2)

# =============================================================================
# ========================= readEmissionNotes =================================
# =============================================================================
def readEmissionTXTnotes(
    fdir: str,
    save: bool = True,
    return_path: bool = False,
) -> 'pd.DataFrame | tuple[pd.DataFrame, list[str]]':
    """
    Parse a legacy TXT-format emission measurement notes file.

    Locates a single ``.txt`` file in *fdir* (or uses *fdir* directly as a
    file path), extracts per-sample metadata, and returns a DataFrame with
    the same columns as :func:`readEmissionCSVnotes`.

    Each record is delimited by a ``File name:`` line; all fields are parsed
    by extracting the first numeric token after the colon, so unit strings
    (``deg C``, ``ohms``) and minor typos (e.g. ``80.4.``) are tolerated.
    Time fields accept both 12-hour (``2:17 PM``) and bare (``2:35``) formats.

    Parameters
    ----------
    fdir : str
        Directory to search for a single ``.txt`` notes file, or the path
        to the file itself.
    save : bool
        If True, save the parsed DataFrame as a CSV alongside the TXT file.
    return_path : bool
        If True, return ``(DataFrame, [file_paths])`` instead of just the
        DataFrame.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, list[str]]
        Parsed notes with columns ``sample_name``, ``dtime``,
        ``channel_101`` … ``channel_107``.  If *return_path* is True, also
        returns the list of file paths involved.
    """
    from datetime import datetime

    if os.path.isdir(fdir):
        logging.info("Working in directory: %s", fdir)
        flist = findFiles("", ".txt", fdir)
        if len(flist) == 0:
            raise IOError("No TXT file found in folder %s" % fdir)
        elif len(flist) > 1:
            raise IOError("Found multiple TXT files in folder %s" % fdir)
        fname = flist[0]
    elif os.path.exists(fdir):
        fname = fdir
        flist = [fname]
    else:
        raise IOError("Path does not exist: %s" % fdir)

    logging.info("Found emission note file: %s", fname)

    def _parse_float(val_str: str) -> float:
        """Extract the first numeric value from a field string."""
        m = re.search(r'\d+\.?\d*', val_str)
        return float(m.group()) if m else np.nan

    def _parse_time(val_str: str) -> 'time | None':
        """Parse HH:MM, H:MM, or H:MM AM/PM from the value side of a Time: line."""
        s = val_str.strip()
        if not s:
            return None
        for fmt in ('%I:%M %p', '%I:%M:%S %p', '%H:%M:%S', '%H:%M'):
            try:
                return datetime.strptime(s, fmt).time()
            except ValueError:
                continue
        return None

    _BLANK_RECORD: dict = {
        'sample_name': '',
        'dtime':       None,
        'channel_101': np.nan,
        'channel_102': np.nan,
        'channel_103': np.nan,
        'channel_104': np.nan,
        'channel_105': np.nan,
        'channel_106': np.nan,
        'channel_107': np.nan,
    }

    records: list[dict] = []
    current: dict | None = None

    with open(fname, 'r') as f:
        lines = f.readlines()

    for raw_line in lines:
        line = raw_line.rstrip('\n')
        line_lower = line.lower()

        if line_lower.lstrip().startswith('file name:'):
            if current is not None:
                records.append(current)
            current = dict(_BLANK_RECORD)
            current['sample_name'] = line.split(':', 1)[-1].strip()
            continue

        if current is None:
            continue

        if line_lower.lstrip().startswith('time:'):
            current['dtime'] = _parse_time(line.split(':', 1)[-1])
            continue

        m = re.search(r'channel\s+(\d+)', line_lower)
        if m:
            ch_num = int(m.group(1))
            val_str = line.split(':', 1)[-1]
            col = f'channel_10{ch_num}' if ch_num <= 9 else f'channel_1{ch_num}'
            if col in current:
                current[col] = _parse_float(val_str)

    if current is not None:
        records.append(current)

    df = pd.DataFrame(records)

    if save:
        csv_fname = fname.replace('.txt', '.csv')
        if csv_fname not in flist:
            flist.append(csv_fname)
        df.to_csv(csv_fname, index=False)
        logging.info("Saved CSV note file: %s", csv_fname)

    if return_path:
        return df, flist
    return df


def readEmissionCSVnotes(
    fpath: str,
    return_path: bool = False,
) -> 'pd.DataFrame | tuple[pd.DataFrame, list[str]]':
    """
    Load an emission measurement notes file (CSV, XLS, or XLSX).

    Parameters
    ----------
    fpath : str
        Path to a notes file directly (``.csv`` / ``.xls`` / ``.xlsx``), or
        a directory in which a single ``*info*`` file with one of those
        extensions will be located automatically.
    return_path : bool
        If True, return ``(DataFrame, [filepath])`` instead of just the
        DataFrame.

    Returns
    -------
    pd.DataFrame or tuple[pd.DataFrame, list[str]]
        Parsed measurement notes.  If *return_path* is True, also returns
        a one-element list containing the resolved file path.

    Raises
    ------
    IOError
        If the path does not exist, no matching file is found, multiple
        candidates are found, or the extension is not supported.
    """
    _SUPPORTED = ('.csv', '.xls', '.xlsx')

    if os.path.isdir(fpath):
        logging.info("Working in directory: %s", fpath)
        flist = findFiles('info', list(_SUPPORTED), fpath)
        if len(flist) == 0:
            raise IOError(
                "No info file (.csv/.xls/.xlsx) found in folder %s" % fpath
            )
        if len(flist) > 1:
            # Nextcloud (and other sync clients) may create conflict copies
            # alongside the original, e.g. "foo (conflicted copy ...).csv".
            # Strip those out before deciding whether the result is ambiguous.
            clean = [f for f in flist
                     if 'conflicted' not in os.path.basename(f).lower()
                     and 'conflict'  not in os.path.basename(f).lower()]
            if len(clean) == 1:
                logging.warning(
                    "Ignoring %d conflict copy/copies alongside '%s'",
                    len(flist) - 1, clean[0],
                )
                flist = clean
            elif len(clean) == 0:
                # All copies are conflict files — take the most recently modified.
                flist = [max(flist, key=os.path.getmtime)]
                logging.warning(
                    "All info files appear to be conflict copies; "
                    "using most recently modified: %s", flist[0],
                )
            else:
                raise IOError(
                    "Found multiple info files in folder %s: %s" % (fpath, clean)
                )
        fname = flist[0]
    elif os.path.isfile(fpath):
        fname = fpath
    else:
        raise IOError("Path does not exist: %s" % fpath)

    ext = os.path.splitext(fname)[1].lower()
    if ext not in _SUPPORTED:
        raise IOError(
            "Unsupported file extension '%s'. Expected one of %s" % (ext, _SUPPORTED)
        )

    logging.info("Found measurement info file: %s", fname)

    if ext == '.csv':
        df = pd.read_csv(fname)
    else:
        df = pd.read_excel(fname)

    if return_path:
        return df, [fname]
    return df


# HDF Tools
def printStructInfo(data_dict: dict, level: int | None = None) -> None:
    """
    Recursively print the structure of a nested dictionary to stdout.

    Parameters
    ----------
    data_dict : dict
        Nested dictionary to inspect.
    level : int or None
        Current indentation depth.  Pass None (default) to start at the
        root level.
    """
    for key, item in data_dict.items():
        if level is None:
            level = 0
        prefix = "\t" * level
        if isinstance(item, dict):
            print(f"{prefix}{key}:")
            printStructInfo(item, level=level + 1)
        elif isinstance(item, np.ndarray):
            print(f"{prefix}{key: <20}: array of size {item.shape} | type: {item.dtype}")
        elif isinstance(item, str):
            print(f"{prefix}{key: <20}: string | '{item}'")


def find_key_recursively(key_to_find: str, dictionary: dict | list) -> Iterator:
    """
    Recursively yield all values for a key in a nested dict or list.

    Parameters
    ----------
    key_to_find : str
        Key to search for at any depth.
    dictionary : dict or list
        Nested structure to search.

    Yields
    ------
    object
        Each value found at *key_to_find* anywhere in the structure.
    """
    if isinstance(dictionary, dict):
        for k, v in dictionary.items():
            if k == key_to_find:
                yield v
            # Recurse if the value is a dict or a list
            if isinstance(v, (dict, list)):
                yield from find_key_recursively(key_to_find, v)
    elif isinstance(dictionary, list):
        # Recurse for each item in the list
        for item in dictionary:
            yield from find_key_recursively(key_to_find, item)


def _read_hdf_raw(h5_object: h5py.File, path: str = '/') -> dict:
    """
    Recursively load all datasets and groups from an HDF5 object into a dict.

    Handles three dataset forms: regular ndarrays (squeezed), scalars
    (int/float/str/bytes), and DaVinci-packed byte strings (a single bytes
    value containing all items joined by ``\\n``).  Byte strings are decoded
    to UTF-8.  Big-endian floats and ints are cast to native equivalents.

    **DaVinci BSQ layout fix** — datasets tagged ``org=0`` (BSQ) come in two
    variants depending on whether the DaVinci writer reversed the dimension
    order for HDF5:

    * *Reversed-shape* variant: HDF5 shape ``(n_spec, 1, n_pts)`` but the flat
      buffer was written in IDL column-major ``(n_pts, 1, n_spec)`` order.
      Identified by the spectral length being the *last* HDF5 dimension
      (``xaxis.shape[-1] > xaxis.shape[0]``).  Fix: ravel + reshape with
      reversed dimensions.
    * *Natural-shape* variant: HDF5 shape ``(n_pts, 1, n_spec)`` with the flat
      buffer in matching column-major order.  No fix needed;
      ``_normalize_hdf_group`` will transpose the spectral axis to last.

    Parameters
    ----------
    h5_object : h5py.File or h5py.Group
        Open HDF5 file or group handle.
    path : str
        HDF5 path to start traversal from.

    Returns
    -------
    dict
        Nested dictionary mirroring the HDF5 group/dataset hierarchy.
    """
    data_dict = {}

    for key, item in h5_object[path].items():

        # If it's a dataset, read the data into a numpy array
        if isinstance(item, h5py.Dataset):

            val = item[()]

            # DaVinci BSQ layout fix (see docstring for details).
            if (isinstance(val, np.ndarray) and val.ndim == 3
                    and 'org' in item.attrs):
                try:
                    xax_shape = h5_object[path + 'xaxis'].shape
                    needs_reversal = xax_shape[-1] > xax_shape[0]
                except KeyError:
                    # No sibling xaxis; fall back to data shape heuristic
                    needs_reversal = val.shape[-1] > val.shape[0]
                if needs_reversal:
                    val = val.ravel().reshape(val.shape[::-1])

            # Squeeze ndarrays; keep Python scalars as-is
            if isinstance(val, np.ndarray):
                val = np.squeeze(val)
                # 0-d array → Python scalar (must happen before the bytes branch
                # so that a 0-d byte array becomes a plain `bytes` object)
                if val.ndim == 0:
                    val = val.item()

            # Decode bytes
            if isinstance(val, (bytes, np.bytes_)):
                # Scalar bytes — DaVinci packs multiple values as one \n-joined string
                decoded = val.decode('utf-8')
                if '\n' in decoded:
                    val = np.array([p for p in decoded.split('\n') if p])
                else:
                    val = decoded
            elif isinstance(val, np.ndarray) and np.issubdtype(val.dtype, np.bytes_):
                if val.size == 1:
                    # Single-element array — check for DaVinci packing
                    decoded = val.flat[0].decode('utf-8')
                    if '\n' in decoded:
                        val = np.array([p for p in decoded.split('\n') if p])
                    else:
                        val = decoded
                else:
                    # Regular array of byte strings — decode each element
                    val = val.astype(str)
            elif isinstance(val, np.ndarray) and val.dtype.kind == 'O':
                # Object array — h5py variable-length UTF-8 strings come back this way.
                # Elements are bytes; decode each one.
                if val.size > 0 and isinstance(val.flat[0], (bytes, np.bytes_)):
                    decoded = [
                        item.decode('utf-8') if isinstance(item, (bytes, np.bytes_)) else str(item)
                        for item in val.flat
                    ]
                    val = np.array(decoded, dtype=str).reshape(val.shape)

            # Uniform numeric dtypes for big-endian arrays
            if isinstance(val, np.ndarray):
                if ">f" in str(val.dtype):
                    val = val.astype(float)
                elif ">i" in str(val.dtype):
                    val = val.astype(int)

            # numpy integer/float scalars → Python native (covers values that
            # came from 0-d arrays via .item() as well as direct numpy scalars)
            if isinstance(val, np.integer):
                val = int(val)
            elif isinstance(val, np.floating):
                val = float(val)

            data_dict[key] = val

        elif isinstance(item, h5py.Group):
            data_dict[key] = _read_hdf_raw(h5_object, path + key + '/')

    return data_dict


def saveHDF(data: dict, fname: str) -> None:
    """
    Save a nested dict to an HDF5 file, readable by :func:`readHDF`.

    Dicts are written as HDF5 groups, numpy arrays as datasets, scalars as
    0-d datasets, and strings as byte-encoded datasets.

    Parameters
    ----------
    data : dict
        Nested dictionary to save.  Values may be numpy arrays, scalars,
        strings, or sub-dicts (written as HDF5 groups).
    fname : str
        Output file path (e.g. ``"results.hdf"``).
    """
    def _write_group(h5_group: h5py.Group, d: dict) -> None:
        for key, val in d.items():
            skey = str(key)   # HDF5 names must be strings
            if isinstance(val, dict):
                grp = h5_group.require_group(skey)
                _write_group(grp, val)
            elif isinstance(val, np.ndarray):
                h5_group.create_dataset(skey, data=val)
            elif isinstance(val, str):
                h5_group.create_dataset(skey, data=val.encode('utf-8'))
            elif isinstance(val, (int, float, np.integer, np.floating)):
                h5_group.create_dataset(skey, data=np.array(val))
            elif isinstance(val, list):
                # Encode as a UTF-8 string array if contents are strings,
                # otherwise fall back to a numpy array
                if len(val) > 0 and isinstance(val[0], str):
                    encoded = np.array([s.encode('utf-8') for s in val])
                    h5_group.create_dataset(skey, data=encoded,
                                            dtype=h5py.string_dtype(encoding='utf-8'))
                else:
                    try:
                        h5_group.create_dataset(skey, data=np.array(val))
                    except TypeError:
                        logging.warning(
                            "saveHDF: skipping key '%s' (list with unsupported element type %s)",
                            skey, type(val[0]) if val else 'empty',
                        )
            else:
                try:
                    h5_group.create_dataset(skey, data=np.array(val))
                except TypeError:
                    logging.warning("saveHDF: skipping key '%s' (unsupported type %s)", skey, type(val))

    with h5py.File(fname, 'w') as f:
        _write_group(f, data)


def save_emcal_csv(out: dict, fname: str) -> None:
    """
    Save emissivity spectra from an emcal output dict to a CSV file.

    The file has one row per wavenumber point.  The first column is the
    wavenumber axis (cm⁻¹); subsequent columns are the retrieved emissivity
    spectra, one per sample, in the order given by ``out['label']``.  Column
    headers are ``wavenumber_cm-1`` followed by the sample labels.

    Parameters
    ----------
    out : dict
        Output dict returned by :func:`~functions.emcal` or :func:`~functions.merge`.
        Required keys: ``xaxis``, ``emiss``, ``label``.
    fname : str
        Output file path (e.g. ``"emcal_results.csv"``).
    """
    import pandas as pd

    xaxis  = np.asarray(out['xaxis'])
    labels = list(out['label'])
    emiss  = out['emiss']

    df = pd.DataFrame(
        {lbl: emiss[lbl] for lbl in labels},
        index=xaxis,
    )
    df.index.name = 'wavenumber_cm-1'
    df.to_csv(fname)


def save_sma_csv(
    out: dict,
    path: str,
    group: bool = False,
) -> list[str]:
    """
    Save SMA concentration results to one or two wide-format CSV files.

    Each row is one sample.  Columns are: ``sample_label``, ``rms``,
    ``bb_pct``, ``bb_pct_err``, ``bb_normconc``, ``bb_normconc_err``,
    then for every endmember a pair ``{label}`` (normalised concentration %)
    and ``{label}_err`` (normalised error %).  Missing error values are
    written as NaN.

    When ``group=True`` and grouped concentrations are present in *out*, a
    second file with the suffix ``_grouped`` is written alongside the first
    using the same layout but with mineral-group labels.

    Parameters
    ----------
    out : dict
        Output dict from :func:`~functions.sma`.
    path : str
        Output file path.  Must end in ``.csv``; if it does not, the
        extension is appended automatically.
    group : bool
        If True, also write a grouped-concentration CSV (requires
        ``out['grouped']``).

    Returns
    -------
    list[str]
        Paths of all files written.
    """
    import pandas as pd

    if not path.endswith('.csv'):
        path = path + '.csv'

    written: list[str] = []

    def _build_df(
        sample_labels: list[str],
        labels: list[str],
        normconc: np.ndarray,
        normerror: np.ndarray | None,
        bb: np.ndarray,
        bberror: np.ndarray | None,
        bb_normconc: np.ndarray,
        bb_normconc_err: np.ndarray | None,
        sl: np.ndarray,
        slerror: np.ndarray | None,
        sl_normconc: np.ndarray,
        sl_normconc_err: np.ndarray | None,
        rms: np.ndarray,
    ) -> 'pd.DataFrame':
        rows: list[dict] = []
        for i, lbl in enumerate(sample_labels):
            row: dict = {
                'sample_label':       lbl,
                'rms':                float(rms[i]),
                'bb_pct':             float(bb[i]) * 100.0,
                'bb_pct_err':         float(bberror[i]) * 100.0 if bberror is not None else float('nan'),
                'bb_normconc':        float(bb_normconc[i]),
                'bb_normconc_err':    float(bb_normconc_err[i]) if bb_normconc_err is not None else float('nan'),
                'slope_pct':          float(sl[i]) * 100.0,
                'slope_pct_err':      float(slerror[i]) * 100.0 if slerror is not None else float('nan'),
                'slope_normconc':     float(sl_normconc[i]),
                'slope_normconc_err': float(sl_normconc_err[i]) if sl_normconc_err is not None else float('nan'),
            }
            for j, em in enumerate(labels):
                row[em]          = float(normconc[i, j])
                row[f'{em}_err'] = float(normerror[i, j]) if normerror is not None else float('nan')
            rows.append(row)
        return pd.DataFrame(rows)

    # ── Per-endmember CSV ────────────────────────────────────────────────────
    sample_labels = out.get('sample_labels', [])
    n_samples_csv = len(sample_labels)
    labels        = out.get('labels', [])
    normconc      = np.asarray(out['normconc'])
    normerror     = np.asarray(out['normerror']) if 'normerror' in out else None
    bb            = np.asarray(out['bb'])
    bberror       = np.asarray(out['bberror'])    if 'bberror'    in out else None
    sl            = np.asarray(out.get('slope', np.zeros(n_samples_csv)))
    slerror       = np.asarray(out['slopeerror']) if 'slopeerror' in out else None
    bb_normconc   = np.asarray(out['bb_normconc'])
    sl_normconc   = np.asarray(out.get('slope_normconc', np.zeros(n_samples_csv)))
    rms           = np.asarray(out['rms'])
    mineral_conc  = np.asarray(out['conc'])

    # bb_normconc_err: bberror / grand_total * 100
    if bberror is not None:
        grand_total_frac = mineral_conc.sum(axis=1) + bb + sl
        bb_normconc_err = np.where(
            grand_total_frac > 0, bberror / grand_total_frac * 100.0, 0.0
        )
    else:
        bb_normconc_err = None

    # slope_normconc_err: slerror / (minerals + slope) * 100
    if slerror is not None:
        remainder_frac = mineral_conc.sum(axis=1) + sl
        sl_normconc_err = np.where(
            remainder_frac > 0, slerror / remainder_frac * 100.0, 0.0
        )
    else:
        sl_normconc_err = None

    df = _build_df(
        sample_labels, labels,
        normconc, normerror,
        bb, bberror,
        bb_normconc, bb_normconc_err,
        sl, slerror,
        sl_normconc, sl_normconc_err,
        rms,
    )
    df.to_csv(path, index=False, float_format='%.4f')
    logging.info("Saved SMA results → %s", path)
    written.append(path)

    # ── Grouped CSV (optional) ───────────────────────────────────────────────
    if group:
        gp = out.get('grouped')
        if gp is None:
            logging.warning("save_sma_csv: group=True but 'grouped' key absent — skipping.")
        else:
            g_labels   = gp['grouped_labels']
            g_conc     = np.asarray(gp['grouped_conc'])
            g_normconc = np.asarray(gp['grouped_normconc'])
            g_bb       = np.asarray(gp.get('grouped_bb', bb))
            g_sl       = np.asarray(gp.get('grouped_slope', np.zeros(n_samples_csv)))
            g_bberror  = np.asarray(gp['grouped_bberror'])    if 'grouped_bberror'    in gp else None
            g_slerror  = np.asarray(gp['grouped_slopeerror']) if 'grouped_slopeerror' in gp else None

            if 'grouped_normerror' in gp:
                g_normerror = np.asarray(gp['grouped_normerror'])
            elif 'grouped_error' in gp:
                g_raw_err   = np.asarray(gp['grouped_error'])
                g_sum       = g_conc.sum(axis=-1, keepdims=True)
                g_normerror = np.where(g_sum > 0, g_raw_err / g_sum * 100.0, 0.0)
            else:
                g_normerror = None

            # BB: normalised relative to grand_total (mineral groups + BB + slope)
            g_grand_total = g_conc.sum(axis=-1) + g_bb + g_sl
            g_bb_normconc = np.where(g_grand_total > 0, g_bb / g_grand_total * 100.0, 0.0)
            g_bb_normconc_err = (
                np.where(g_grand_total > 0, g_bberror / g_grand_total * 100.0, 0.0)
                if g_bberror is not None else None
            )

            # Slope: normalised relative to (mineral groups + slope)
            g_remainder  = g_conc.sum(axis=-1) + g_sl
            g_sl_normconc = np.where(g_remainder > 0, g_sl / g_remainder * 100.0, 0.0)
            g_sl_normconc_err = (
                np.where(g_remainder > 0, g_slerror / g_remainder * 100.0, 0.0)
                if g_slerror is not None else None
            )

            group_path = path.replace('.csv', '_grouped.csv')
            gdf = _build_df(
                sample_labels, g_labels,
                g_normconc, g_normerror,
                g_bb, g_bberror,
                g_bb_normconc, g_bb_normconc_err,
                g_sl, g_slerror,
                g_sl_normconc, g_sl_normconc_err,
                rms,
            )
            gdf.to_csv(group_path, index=False, float_format='%.4f')
            logging.info("Saved grouped SMA results → %s", group_path)
            written.append(group_path)

    return written


def _detect_hdf_format(fname: str) -> str:
    """
    Inspect the top-level structure of an HDF5 file and return a format tag.

    Returns
    -------
    str
        ``'flat'``         – top level has ``data`` or ``spectra`` + ``xaxis``
                             (DaVinci native or makeASUspeclib grouped format).
        ``'per_spectrum'`` – top-level keys are all numeric strings and each is
                             a group containing ``data`` + ``xaxis``
                             (SpeclibViewerTIR / saveHDF export format).
    """
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
        has_data_key = any(k in keys for k in ('data', 'spectra'))
        has_xaxis    = 'xaxis' in keys
        all_numeric  = all(k.lstrip('-').isdigit() for k in keys)
        first_is_grp = keys and isinstance(f[keys[0]], h5py.Group)

    if has_data_key and has_xaxis:
        return 'flat'
    if all_numeric and first_is_grp:
        return 'per_spectrum'
    # Default: let the existing recursive reader handle it
    return 'flat'


def _collapse_per_spectrum(d: dict) -> dict:
    """
    Collapse a per-spectrum HDF dict (one sub-dict per spectrum) into a flat
    ``{data, xaxis, label, <metadata arrays>}`` structure matching the output
    of the flat DaVinci format.

    Sub-dict keys that hold scalar strings or numbers are collected into 1-D
    arrays.  ``data`` arrays are stacked into ``(n_spectra, n_bands)``.

    Raises ``ValueError`` if the per-spectrum xaxes are not all identical
    (checked with ``np.allclose``), since stacking spectra with mismatched grids
    is undefined.  In that case, call ``readHDF`` with ``collapse=False`` to
    receive the raw per-spectrum dict instead.

    Parameters
    ----------
    d : dict
        Top-level dict from ``_read_hdf_raw`` for a per-spectrum file.

    Returns
    -------
    dict
        Flat structure with ``data`` (n_spectra, n_bands), ``xaxis`` (n_bands,),
        ``label`` (alias of ``sample_name`` if present), and one array per
        collected metadata field.
    """
    # Sort numerically so spectrum order is preserved
    spec_keys = sorted(d.keys(), key=lambda k: int(k))
    entries   = [d[k] for k in spec_keys]

    # Validate that all xaxes are identical before stacking
    xaxis = entries[0]['xaxis']
    for i, e in enumerate(entries[1:], start=1):
        ex = e['xaxis']
        if ex.shape != xaxis.shape or not np.allclose(ex, xaxis):
            raise ValueError(
                f"readHDF: per-spectrum entry {spec_keys[i]} has a different "
                f"xaxis (shape {ex.shape}) from entry {spec_keys[0]} "
                f"(shape {xaxis.shape}). Cannot collapse to flat format. "
                f"Call readHDF(..., collapse=False) to load as a per-spectrum dict."
            )

    data = np.stack([e['data'] for e in entries], axis=0)   # (n_spectra, n_bands)
    out: dict = {'data': data, 'xaxis': xaxis}

    # Collect scalar metadata fields into arrays
    skip = {'data', 'xaxis'}
    meta_keys = [k for k in entries[0] if k not in skip]
    for key in meta_keys:
        vals = [e.get(key, '') for e in entries]
        # Check if all values are scalar (str, int, float)
        if all(isinstance(v, (str, int, float, np.integer, np.floating)) for v in vals):
            out[key] = np.array(vals)

    # Ensure a 'label' key exists (mirrors DaVinci flat format convention)
    if 'label' not in out and 'sample_name' in out:
        out['label'] = out['sample_name']

    return out


def _normalize_hdf_group(d: dict) -> dict:
    """
    Recursively normalise a dict returned by ``_read_hdf_raw``.

    Two transformations are applied at every level that contains ``xaxis``:

    1. **Key rename** – ``spectra`` is renamed to ``data`` when ``data`` is
       absent.  This unifies the key used by DaVinci files (``data``) and the
       clean grouped format written by ``makeASUspeclib_dev`` (``spectra``).

    2. **Axis rearrangement** – any array whose shape contains a dimension
       equal to ``len(xaxis)`` has that dimension moved to the last position,
       so the spectral axis is always the trailing axis (consistent with the
       convention ``(..., n_pts)``).

    Levels without ``xaxis`` are traversed to find nested groups that do
    have one (e.g. the root level of a grouped speclib file whose xaxis lives
    inside each sub-group).

    Parameters
    ----------
    d : dict
        One level of the nested dict from ``_read_hdf_raw``.

    Returns
    -------
    dict
        The same dict, mutated in-place and returned.
    """
    if 'xaxis' in d:
        # Rename spectra → data for the makeASUspeclib_dev clean format
        if 'spectra' in d and 'data' not in d:
            d['data'] = d.pop('spectra')

        # Squeeze xaxis to 1D: DaVinci BSQ stores it as (1, 1, n_pts) with
        # degenerate spatial dimensions.  len() on a 3-D array returns the
        # size of the first axis, not the spectral length.
        xaxis = np.atleast_1d(d['xaxis'].squeeze())
        d['xaxis'] = xaxis
        nx = len(xaxis)

        for key, val in list(d.items()):
            if key == 'xaxis':
                continue
            if isinstance(val, dict):
                d[key] = _normalize_hdf_group(val)
            elif isinstance(val, np.ndarray):
                ix = [i for i, s in enumerate(val.shape) if s == nx]
                if len(ix) == 1:
                    # Move spectral dim to last, then drop any remaining size-1 dims
                    val = np.moveaxis(val, source=ix[0], destination=-1)
                    squeeze_axes = tuple(
                        i for i, s in enumerate(val.shape[:-1]) if s == 1
                    )
                    if squeeze_axes:
                        val = val.squeeze(axis=squeeze_axes)
                    d[key] = val
                elif len(ix) > 1:
                    raise RuntimeError(
                        f"readHDF: key '{key}' has more than one dimension "
                        f"matching xaxis length {nx} — cannot determine spectral axis"
                    )
                elif val.ndim > 1:
                    # No spectral dim — metadata array with degenerate dims; squeeze
                    val = val.squeeze()
                    if val.ndim == 0:
                        val = val.reshape(1)
                    d[key] = val
    else:
        # No xaxis at this level; recurse into sub-dicts
        for key, val in list(d.items()):
            if isinstance(val, dict):
                d[key] = _normalize_hdf_group(val)

    return d


def readHDF(
    fname: str,
    path: str = '/',
    printout: bool = False,
    collapse: bool = True,
) -> dict:
    """
    Open an HDF5 file and load its contents into a nested dict.

    Handles three on-disk formats transparently:

    * **DaVinci native** – big-endian numeric arrays, ``\\n``-packed byte
      strings, ``data`` key holding spectra in ``(n_pts, n_spec)`` order.
    * **Grouped clean format** (``makeASUspeclib_dev``) – UTF-8 variable-length
      strings, ``spectra`` key, spectra in ``(n_spec, n_pts)`` order.
    * **Per-spectrum format** (:func:`saveHDF` / SpeclibViewer export) – one
      sub-group per spectrum keyed by a numeric string, ``data`` key shape
      ``(n_pts,)``.

    For the per-spectrum format, behaviour is controlled by ``collapse``:

    * ``collapse=True`` (default) – stack all spectra into a flat
      ``{data (n_spectra, n_bands), xaxis, label, …}`` structure matching the
      other formats.  Raises ``ValueError`` if per-spectrum xaxes differ.
    * ``collapse=False`` – return the raw per-spectrum dict
      ``{'1': {data, xaxis, …}, '2': {…}, …}`` so heterogeneous grids are
      preserved faithfully.

    After loading, spectral axes are normalised recursively: ``spectra`` keys
    are renamed to ``data`` and any array dimension equal to ``len(xaxis)``
    is moved to the last position so the spectral axis is always trailing.

    Parameters
    ----------
    fname : str
        Path to the HDF5 file.
    path : str
        HDF5 path to start traversal from.
    printout : bool
        If True, print a structural summary of the loaded dict.
    collapse : bool
        For per-spectrum format files only.  If True (default), collapse to a
        flat array structure.  Set to False to preserve the per-spectrum dict
        when spectra have heterogeneous xaxes.

    Returns
    -------
    dict
        Flat structure ``{data, xaxis, label, …}`` for flat and collapsed
        per-spectrum formats; per-spectrum dict ``{'1': {…}, …}`` when
        ``collapse=False``.
    """
    fmt = _detect_hdf_format(fname)

    with h5py.File(fname, 'r') as h5_object:
        data_dict = _read_hdf_raw(h5_object, path)

    if fmt == 'per_spectrum':
        if collapse:
            data_dict = _collapse_per_spectrum(data_dict)
    else:
        data_dict = _normalize_hdf_group(data_dict)

    if printout:
        print(f"HDF structure with {len(data_dict.keys())} elements")
        printStructInfo(data_dict)

    return data_dict

def readOMNIC(fname: str) -> dict:
    """
    Read a two-column OMNIC CSV spectrum file.

    Parameters
    ----------
    fname : str
        Path to the CSV file.  Expected format: no header, two columns
        (wavenumber, intensity).

    Returns
    -------
    dict
        ``{'wn': np.ndarray, 'data': np.ndarray}`` with NaN and zero-valued
        rows removed.
    """
    df   = pd.read_csv(fname, header=None, names=['wn', 'data'])
    wn   = df['wn'].to_numpy()
    data = df['data'].to_numpy()

    idx = np.isfinite(wn) & np.isfinite(data) & (wn != 0) & (data != 0)
    return {'wn': wn[idx], 'data': data[idx]}


# =============================================================================
# ========================= Reflectance file I/O ==============================
# =============================================================================

# Column names accepted as the wavelength axis (checked case-insensitively).
_WL_COLUMN_NAMES: frozenset[str] = frozenset({'wavelength', 'wl', 'wav', 'lambda', 'nm'})


def loadReflectanceCSV(path: 'str | Path') -> pd.DataFrame:
    """
    Parse a generic wide-format reflectance CSV into a DataFrame.

    The file must contain exactly one wavelength column, identified
    case-insensitively from :data:`_WL_COLUMN_NAMES`.  All remaining columns
    are treated as individual reflectance spectra.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: wavelength column followed by one column per
        spectrum, with original column names preserved.

    Raises
    ------
    ValueError
        If no wavelength column is found or no spectrum columns are present.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    df = pd.read_csv(path)

    wl_cols = [c for c in df.columns if c.strip().lower() in _WL_COLUMN_NAMES]
    if not wl_cols:
        raise ValueError(
            f"No wavelength column found in '{path.name}'.\n"
            f"Expected a column named one of: {', '.join(sorted(_WL_COLUMN_NAMES))}."
        )
    if not any(c != wl_cols[0] for c in df.columns):
        raise ValueError(
            f"'{path.name}' contains only the wavelength column; no spectrum data found."
        )
    return df


def saveReflectanceCSV(
    path:    'str | Path',
    xaxis:   'np.ndarray',
    spectra: 'dict[str, np.ndarray]',
) -> None:
    """
    Write a wide-format reflectance CSV readable by :func:`loadReflectanceCSV`.

    Parameters
    ----------
    path : str or Path
        Destination file path.  Any existing file is overwritten.
    xaxis : np.ndarray
        Wavelength axis, shape ``(n_channels,)``.
    spectra : dict[str, np.ndarray]
        Mapping of spectrum name → 1-D reflectance array, each shape
        ``(n_channels,)``.  Column order follows dict insertion order.

    Raises
    ------
    ValueError
        If *spectra* is empty or any spectrum length does not match *xaxis*.
    """
    from pathlib import Path as _Path
    import numpy as _np
    path = _Path(path)

    if not spectra:
        raise ValueError("'spectra' is empty — nothing to write.")
    for name, data in spectra.items():
        if len(data) != len(xaxis):
            raise ValueError(
                f"Spectrum '{name}' has {len(data)} samples but xaxis has "
                f"{len(xaxis)} — lengths must match."
            )

    pd.DataFrame({'Wavelength': xaxis, **spectra}).to_csv(path, index=False)


def load_reflectance_vswir(
    path: 'str | Path',
    *,
    fmt: str = 'auto',
) -> dict:
    """
    Load a VSWIR reflectance file and return a data dict.

    Thin convenience wrapper around :func:`loadReflectanceCSV` and
    :func:`loadASD` that converts the resulting DataFrame directly to numpy
    arrays, so callers never need to handle the intermediate DataFrame.

    Parameters
    ----------
    path : str or Path
        Path to the reflectance file.
    fmt : {'auto', 'csv', 'asd'}
        File format.  ``'auto'`` infers from the file extension: ``.txt``
        is treated as an ASD export, everything else as CSV.

    Returns
    -------
    dict
        ``'xaxis'``  : np.ndarray, shape (n_channels,) — wavelength axis in nm.
        ``'spectra'``: dict[str, np.ndarray] — spectrum name → 1-D array.
        ``'source'`` : Path — resolved file path.

    Raises
    ------
    ValueError
        Propagated from :func:`loadReflectanceCSV` / :func:`loadASD` on
        missing wavelength column or non-numeric spectrum columns.
    """
    from pathlib import Path as _Path
    path = _Path(path)

    if fmt == 'auto':
        fmt = 'asd' if path.suffix.lower() == '.txt' else 'csv'

    df = loadASD(path) if fmt == 'asd' else loadReflectanceCSV(path)

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

    return {'xaxis': xaxis, 'spectra': spectra, 'source': path}


def loadASD(path: 'str | Path') -> pd.DataFrame:
    """
    Parse an ASD spectrometer tab-separated text export into a DataFrame.

    The ASD ViewSpecPro export produces a wide-format table: first column is
    the wavelength axis (nm, integer), remaining columns are named after the
    original ``.asd`` acquisition filenames.

    Parameters
    ----------
    path : str or Path
        Path to the ASD ``.txt`` export file.

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame: wavelength column followed by one column per
        spectrum, with original ASD filenames as column headers.

    Raises
    ------
    ValueError
        If no wavelength column is found or no spectrum columns are present.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    df = pd.read_csv(path, sep='\t')

    wl_cols = [c for c in df.columns if c.strip().lower() in _WL_COLUMN_NAMES]
    if not wl_cols:
        raise ValueError(
            f"No wavelength column found in '{path.name}'.\n"
            f"Expected first column named one of: {', '.join(sorted(_WL_COLUMN_NAMES))}."
        )
    if not any(c != wl_cols[0] for c in df.columns):
        raise ValueError(
            f"'{path.name}' contains only the wavelength column; no spectrum data found."
        )
    return df


# =============================================================================
# Spectral library format conversion
# =============================================================================

def _decode(v: object) -> str:
    """Decode bytes/np.bytes_ to str; pass str through."""
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8', errors='replace')
    return str(v)


def _native_scalar(v: object) -> object:
    """
    Convert a numpy 0-d array or numpy scalar to a Python native type.
    Bytes are decoded to str.
    """
    if isinstance(v, np.ndarray):
        s = v.squeeze()
        raw = s.item() if s.ndim == 0 else s.flat[0]
        if hasattr(raw, 'item'):
            raw = raw.item()
        if isinstance(raw, (bytes, np.bytes_)):
            return raw.decode('utf-8', errors='replace')
        return raw
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode('utf-8', errors='replace')
    return v


def _unpack_string_field(val: np.ndarray, n_spec: int) -> list[str]:
    """
    Expand a string metadata field from a flat DaVinci HDF5 into a
    per-spectrum list of length *n_spec*.

    Two encodings are handled:

    * **Packed** – shape ``(1,)`` byte array whose single element is a
      newline-delimited string with exactly ``n_spec`` lines.
    * **Per-spec** – shape ``(n_spec,)`` array of byte strings.

    Falls back to repeating the first element if the line count does not
    match *n_spec*.
    """
    if not isinstance(val, np.ndarray):
        return [_decode(val)] * n_spec

    if val.dtype.kind not in ('S', 'O', 'U'):
        return [str(x) for x in val[:n_spec]] if len(val) >= n_spec else [str(val.flat[0])] * n_spec

    if len(val) == 1:
        raw = _decode(val[0])
        parts = raw.rstrip('\n').split('\n')
        if len(parts) == n_spec:
            return parts
        logging.warning(
            "_unpack_string_field: packed string has %d lines, expected %d; "
            "repeating first element",
            len(parts), n_spec,
        )
        return [parts[0] if parts else ''] * n_spec

    if len(val) == n_spec:
        return [_decode(x) for x in val]

    logging.warning(
        "_unpack_string_field: array length %d does not match n_spec=%d; "
        "repeating first element",
        len(val), n_spec,
    )
    return [_decode(val.flat[0])] * n_spec


def _flat__to_album(raw: dict) -> dict:
    """
    Convert a flat DaVinci dict (one key per field, ``data`` is 2-D) to
    an album dict ``{i: {'data': ..., 'xaxis': ..., ...}}``.
    """
    xaxis  = np.asarray(raw['xaxis'], dtype=np.float64)
    data   = np.atleast_2d(np.asarray(raw['data'],  dtype=np.float64))
    n_spec = data.shape[0]

    expanded: dict[str, list] = {}
    for key, val in raw.items():
        if key in ('data', 'xaxis'):
            continue
        if not isinstance(val, np.ndarray):
            expanded[key] = [_native_scalar(val)] * n_spec
            continue
        if val.dtype.kind in ('S', 'O', 'U'):
            expanded[key] = _unpack_string_field(val, n_spec)
        elif len(val) == n_spec:
            expanded[key] = [_native_scalar(v) for v in val]
        else:
            expanded[key] = [_native_scalar(val)] * n_spec

    album: dict = {}
    for i in range(n_spec):
        entry: dict = {'data': data[i], 'xaxis': xaxis}
        for key, vals in expanded.items():
            entry[key] = vals[i]
        album[i] = entry
    return album


def _grouped_sv_to_album(raw: dict) -> dict:
    """
    Convert a grouped SV dict (sub-dicts each holding shared ``xaxis`` +
    2-D ``data``) to an album dict.
    """
    album: dict = {}
    next_id = 0
    for grp in raw.values():
        if not isinstance(grp, dict) or 'xaxis' not in grp or 'data' not in grp:
            continue
        xaxis  = np.asarray(grp['xaxis'], dtype=np.float64)
        data   = np.atleast_2d(np.asarray(grp['data'],  dtype=np.float64))
        n_spec = data.shape[0]
        for i in range(n_spec):
            entry: dict = {'data': data[i], 'xaxis': xaxis}
            for key, val in grp.items():
                if key in ('data', 'xaxis'):
                    continue
                if isinstance(val, np.ndarray) and len(val) == n_spec:
                    entry[key] = _native_scalar(val[i])
                else:
                    entry[key] = _native_scalar(val)
            album[next_id + i] = entry
        next_id += n_spec
    return album


def _per_spectrum_to_album(raw: dict) -> dict:
    """
    Convert a per-spectrum dict (one sub-dict per spectrum, keyed by
    spec-id or integer string) to a sequential-integer album dict.
    """
    album: dict = {}
    for new_id, entry in enumerate(raw.values()):
        album[new_id] = {
            k: (val if k in ('data', 'xaxis') else _native_scalar(val))
            for k, val in entry.items()
        }
    return album


def _to_album(raw: dict) -> dict:
    """
    Convert the output of :func:`readHDF` to the per-spectrum album dict
    ``{i: {'data': np.ndarray, 'xaxis': np.ndarray, ...}}`` expected by
    :func:`~speclab.functions.sma` and SpeclibViewer.

    Intended for spectral library structures only — not general HDF outputs
    (e.g. emcal results).  Three layouts are handled automatically:

    * **Flat** — top-level keys are field names; ``data`` is 2-D.
    * **Grouped** — sub-dicts each hold a shared ``xaxis`` + 2-D ``data``.
    * **Per-spectrum** — one sub-dict per spectrum.

    Parameters
    ----------
    raw : dict
        Dict returned by :func:`readHDF` for a spectral library file.

    Returns
    -------
    dict
        Album dict with sequential integer keys starting at 0.

    Raises
    ------
    ValueError
        If the layout cannot be recognised.
    """
    if 'xaxis' in raw and 'data' in raw and not isinstance(raw['data'], dict):
        data_val = raw['data']
        if isinstance(data_val, np.ndarray) and data_val.ndim >= 2:
            return _flat__to_album(raw)
        return {0: {k: v for k, v in raw.items()}}

    if all(isinstance(v, dict) for v in raw.values()):
        first = next(iter(raw.values()))
        if 'xaxis' in first and 'data' in first and np.ndim(first['data']) == 2:
            return _grouped_sv_to_album(raw)
        return _per_spectrum_to_album(raw)

    raise ValueError(
        "_to_album: unrecognised spectral library layout — "
        f"top-level keys: {list(raw.keys())[:8]}"
    )


def save_tracal_csv(out: dict, fname: str) -> None:
    """
    Save transmittance, absorbance, and optical depth spectra from a
    :func:`~functions.tracal` output dict to a single CSV file.

    One row per wavenumber point.  The index column is ``wavenumber_cm-1``.
    Columns are grouped by quantity, each sample suffixed with ``_tra``,
    ``_abs``, or ``_od``.  The mean and standard deviation across samples are
    appended as ``mean_tra`` / ``std_tra``, etc.

    Parameters
    ----------
    out : dict
        Output dict from :func:`~functions.tracal`.
        Required keys: ``wn``, ``tra``.  Optional: ``abs``, ``od``.
    fname : str
        Output file path (e.g. ``"tracal_results.csv"``).
    """
    wn     = np.asarray(out['wn'])
    labels = out['header']['sample_labels']
    cols   = {}

    for key, suffix in (('tra', '_tra'), ('abs', '_abs'), ('od', '_od')):
        if key not in out:
            continue
        block = out[key]
        for lbl in labels:
            cols[lbl + suffix] = block[lbl]
        cols['mean' + suffix] = block['mean']
        cols['std'  + suffix] = block['std']

    df = pd.DataFrame(cols, index=wn)
    df.index.name = 'wavenumber_cm-1'
    df.to_csv(fname)


def save_refcal_csv(out: dict, fname: str) -> None:
    """
    Save reflectance spectra from a :func:`~functions.refcal` output dict
    to a single CSV file.

    One row per wavenumber point.  The index column is ``wavenumber_cm-1``.
    Per-sample columns are suffixed with ``_ref``; the cross-sample mean and
    standard deviation are written as ``mean_ref`` and ``std_ref``.

    Parameters
    ----------
    out : dict
        Output dict from :func:`~functions.refcal`.
        Required keys: ``wn``, ``ref``.
    fname : str
        Output file path (e.g. ``"refcal_results.csv"``).
    """
    wn     = np.asarray(out['wn'])
    labels = out['header']['sample_labels']
    block  = out['ref']

    cols = {lbl + '_ref': block[lbl] for lbl in labels}
    cols['mean_ref'] = block['mean']
    cols['std_ref']  = block['std']

    df = pd.DataFrame(cols, index=wn)
    df.index.name = 'wavenumber_cm-1'
    df.to_csv(fname)


# =============================================================================
# Band parameters CSV I/O
# =============================================================================

# Canonical metric key order — matches functions.band_parameters() return dict.
_BP_METRIC_KEYS: list[str] = [
    'wl_center', 'wl_min', 'band_depth', 'fwhm', 'base_width',
    'band_area', 'band_area_ratio', 'asymmetry_hw', 'asymmetry_centroid',
]
# Wavelength-valued metrics that must be scaled when unit != 'nm'.
_BP_METRIC_SCALED: frozenset[str] = frozenset({
    'wl_center', 'wl_min', 'fwhm', 'base_width', 'band_area',
})

# Magic string that identifies the new CSV format.
_BP_MAGIC       = '# speclab:band_parameters'
_BP_UNIT_PREFIX = '# unit: '
_BP_FEAT_PREFIX = '# features: '

# Column-suffix → (metric_key, unit_hint) for the *old* GUI export format.
# Sorted longest-first so the matching loop picks the most specific suffix.
_BP_OLD_SUFFIXES: list[tuple[str, str, 'str | None']] = sorted([
    ('Band center (nm)',  'wl_center',          'nm'),
    ('Band center (µm)', 'wl_center',          'µm'),
    ('Band min (nm)',     'wl_min',             'nm'),
    ('Band min (µm)',    'wl_min',             'µm'),
    ('Depth',             'band_depth',         None),
    ('FWHM (nm)',         'fwhm',               'nm'),
    ('FWHM (µm)',        'fwhm',               'µm'),
    ('Base width (nm)',   'base_width',         'nm'),
    ('Base width (µm)',  'base_width',         'µm'),
    ('Band area (nm)',    'band_area',          'nm'),
    ('Band area (µm)',   'band_area',          'µm'),
    ('Area ratio',        'band_area_ratio',    None),
    ('Asym HW',           'asymmetry_hw',       None),
    ('Asym centroid',     'asymmetry_centroid', None),
], key=lambda x: -len(x[0]))


def save_band_parameters_csv(
    out:  dict,
    path: 'str | Path',
    unit: str = 'nm',
) -> None:
    """
    Save band-parameter results to a round-trippable CSV file.

    The file begins with three comment rows that store the file format
    identifier, the wavelength unit, and a JSON-encoded feature list (name,
    shoulder window, group).  The data section uses
    ``{feature_name}::{metric_key}`` column names so the file can be loaded
    back without ambiguity.

    Parameters
    ----------
    out : dict
        Output dict from :func:`~functions.band_parameters_batch`.  Required
        keys: ``'features'`` (list of feature dicts) and ``'results'``
        (mapping spectrum name → feature name → band-parameter dict or
        ``None``).  Optional key: ``'sources'`` (mapping spectrum name →
        ``'Data'`` or ``'Library'``).
    path : str or Path
        Destination CSV file path.
    unit : str
        Wavelength unit for saved values.  ``'nm'`` (default) writes raw
        nanometre values; ``'µm'`` divides all wavelength-valued metrics by
        1000.

    Notes
    -----
    ``None`` band-parameter results (features absent from a spectrum) are
    written as empty cells.  ``band_depth`` is used as the presence
    discriminator on load: an empty ``band_depth`` cell means ``None``.
    """
    import json
    import csv
    from pathlib import Path as _Path

    path = _Path(path)
    if unit not in ('nm', 'µm'):
        raise ValueError(f"unit must be 'nm' or 'µm', got {unit!r}")

    feat_list: list[dict] = out['features']
    res_dict:  dict       = out['results']
    sources:   dict       = out.get('sources', {})
    scale = 1e-3 if unit == 'µm' else 1.0

    feat_meta: list[dict] = []
    for feat in feat_list:
        wr = feat.get('wl_range')
        feat_meta.append({
            'name':  feat['name'],
            'wl_lo': float(wr[0]) if wr else None,
            'wl_hi': float(wr[1]) if wr else None,
            'group': feat.get('group', ''),
        })

    with open(path, 'w', newline='', encoding='utf-8') as fh:
        fh.write(_BP_MAGIC + '\n')
        fh.write(_BP_UNIT_PREFIX + unit + '\n')
        fh.write(_BP_FEAT_PREFIX + json.dumps(feat_meta, ensure_ascii=False) + '\n')

        writer = csv.writer(fh)
        header = (['Spectrum', 'Source']
                  + [f'{feat["name"]}::{key}'
                     for feat in feat_list
                     for key  in _BP_METRIC_KEYS])
        writer.writerow(header)

        for sp_name, feat_results in res_dict.items():
            row: list = [sp_name, sources.get(sp_name, '')]
            for feat in feat_list:
                bp = feat_results.get(feat['name'])
                if bp is None:
                    row += [''] * len(_BP_METRIC_KEYS)
                else:
                    for key in _BP_METRIC_KEYS:
                        val = bp.get(key)
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            row.append('')
                        else:
                            row.append(val * scale if key in _BP_METRIC_SCALED else val)
            writer.writerow(row)


def load_band_parameters_csv(
    path: 'str | Path',
) -> dict:
    """
    Load band-parameter results from a CSV written by
    :func:`save_band_parameters_csv` or exported from the ReflectanceVSWIR GUI.

    Detects the format automatically:

    * **New format** — file begins with ``# speclab:band_parameters``:
      unit, feature shoulder windows, and groups are fully recovered.
    * **Old GUI export format** — plain header with columns such as
      ``"1400 nm H2O Band center (nm)"``: feature names and metric keys are
      inferred by suffix-matching; ``wl_range`` and ``group`` are set to
      ``None`` / ``''``.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    dict
        Keys:

        ``'features'``
            ``list[dict]`` — each with ``'name'``, ``'wl_range'`` (``None``
            for old-format files), ``'group'``.
        ``'results'``
            ``dict[str, dict[str, dict | None]]`` — spectrum name →
            feature name → :func:`~functions.band_parameters` result or
            ``None``.  All wavelength-valued metrics are in **nm**.
        ``'sources'``
            ``dict[str, str]`` — spectrum name → ``'Data'``,
            ``'Library'``, or ``''``.
    """
    from pathlib import Path as _Path
    path = _Path(path)
    with open(path, encoding='utf-8') as fh:
        first = fh.readline().rstrip('\n')
    if first == _BP_MAGIC:
        return _load_bp_csv_new(path)
    return _load_bp_csv_old(path)


def _load_bp_csv_new(path: 'Path') -> dict:
    """Parse the ``# speclab:band_parameters`` format."""
    import json
    import csv

    unit      = 'nm'
    feat_meta: list[dict] = []

    with open(path, encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            line = line.rstrip('\n')
            if i == 1 and line.startswith(_BP_UNIT_PREFIX):
                unit = line[len(_BP_UNIT_PREFIX):]
            elif i == 2 and line.startswith(_BP_FEAT_PREFIX):
                feat_meta = json.loads(line[len(_BP_FEAT_PREFIX):])
            elif i >= 3:
                break

    # Values in the file are in `unit`; undo scaling to recover nm.
    scale = 1e-3 if unit == 'µm' else 1.0

    features: list[dict] = []
    for fm in feat_meta:
        wl_lo, wl_hi = fm.get('wl_lo'), fm.get('wl_hi')
        features.append({
            'name':     fm['name'],
            'wl_range': (float(wl_lo), float(wl_hi))
                        if (wl_lo is not None and wl_hi is not None) else None,
            'group':    fm.get('group', ''),
        })
    feat_names = [f['name'] for f in features]

    with open(path, encoding='utf-8') as fh:
        for _ in range(3):      # skip the 3 comment rows
            next(fh)
        reader = csv.DictReader(fh)
        rows   = list(reader)

    results: dict[str, 'dict[str, dict | None]'] = {}
    sources: dict[str, str]                      = {}

    for row in rows:
        sp_name          = row['Spectrum']
        sources[sp_name] = row.get('Source', '')
        feat_results: dict[str, 'dict | None'] = {}

        for feat_name in feat_names:
            # band_depth is never NaN in a valid result — use it as sentinel.
            if row.get(f'{feat_name}::band_depth', '') == '':
                feat_results[feat_name] = None
            else:
                bp: dict[str, float] = {}
                for key in _BP_METRIC_KEYS:
                    val_str = row.get(f'{feat_name}::{key}', '')
                    if val_str == '':
                        bp[key] = float('nan')
                    else:
                        raw = float(val_str)
                        bp[key] = raw / scale if key in _BP_METRIC_SCALED else raw
                feat_results[feat_name] = bp

        results[sp_name] = feat_results

    return {'features': features, 'results': results, 'sources': sources}


# =============================================================================
# GUI helpers
# =============================================================================

def _set_window_size(
    root:       'tk.Wm',
    fraction:   float = 0.85,
    min_w:      int   = 1000,
    min_h:      int   = 640,
    fullscreen: bool  = False,
) -> None:
    """
    Set the initial size and position of a tkinter top-level window.

    When *fullscreen* is ``True`` the window is maximised using the best
    available method for the current windowing system.  Otherwise the window
    is sized to *fraction* of the screen in each dimension and centred.
    *min_w* / *min_h* are always applied as the minimum resizable size.

    Parameters
    ----------
    root : tk.Wm
        Any tkinter top-level widget (``tk.Tk`` or ``tk.Toplevel``).
    fraction : float
        Fraction of screen width and height to use when not fullscreen
        (default 0.85).
    min_w : int
        Minimum window width in pixels.
    min_h : int
        Minimum window height in pixels.
    fullscreen : bool
        Maximise the window to fill the screen (default False).
    """
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    if fullscreen:
        ws = root.tk.call('tk', 'windowingsystem')
        if ws == 'win32':
            root.state('zoomed')
        elif ws == 'aqua':              # macOS
            root.geometry(f'{sw}x{sh}+0+0')
        else:                           # X11 / Linux
            root.attributes('-zoomed', True)
    else:
        w = max(min_w, int(sw * fraction))
        h = max(min_h, int(sh * fraction))
        x = (sw - w) // 2
        y = (sh - h) // 2
        root.geometry(f'{w}x{h}+{x}+{y}')

    root.minsize(min_w, min_h)


def _load_bp_csv_old(path: 'Path') -> dict:
    """
    Parse the old ReflectanceVSWIR GUI export format (best-effort).

    Column names are ``"{feature_name} {metric_display_label}"``.  Feature
    names are recovered by stripping the longest matching known metric suffix.
    ``wl_range`` and ``group`` cannot be recovered and are set to ``None``
    and ``''``.
    """
    import csv

    with open(path, encoding='utf-8') as fh:
        reader     = csv.DictReader(fh)
        rows       = list(reader)
        fieldnames = reader.fieldnames or []

    data_cols = [c for c in fieldnames if c not in ('Spectrum', 'Source')]
    if not data_cols:
        raise ValueError(
            f"'{path.name}' has no recognised data columns — expected the old "
            "ReflectanceVSWIR export format or the speclab:band_parameters format."
        )

    detected_unit: 'str | None' = None
    col_map: list[tuple[str, str]] = []    # (feat_name, metric_key) per column

    for col in data_cols:
        for suffix, metric_key, col_unit in _BP_OLD_SUFFIXES:
            if col.endswith(f' {suffix}'):
                col_map.append((col[: -(len(suffix) + 1)], metric_key))
                if col_unit is not None and detected_unit is None:
                    detected_unit = col_unit
                break
        else:
            raise ValueError(
                f"Column '{col}' in '{path.name}' does not match any known "
                "band-parameter metric label; the file may use an unsupported format."
            )

    unit  = detected_unit or 'nm'
    scale = 1e-3 if unit == 'µm' else 1.0

    seen: dict[str, None] = {}
    for feat_name, _ in col_map:
        seen[feat_name] = None
    feat_names = list(seen.keys())

    features: list[dict] = [
        {'name': fn, 'wl_range': None, 'group': ''}
        for fn in feat_names
    ]

    results: dict[str, 'dict[str, dict | None]'] = {}
    sources: dict[str, str]                      = {}

    for row in rows:
        sp_name          = row.get('Spectrum', '')
        sources[sp_name] = row.get('Source', '')
        feat_bp: dict[str, dict] = {fn: {} for fn in feat_names}

        for col, (feat_name, metric_key) in zip(data_cols, col_map):
            val_str = row.get(col, '')
            if val_str == '':
                feat_bp[feat_name][metric_key] = float('nan')
            else:
                raw = float(val_str)
                feat_bp[feat_name][metric_key] = (
                    raw / scale if metric_key in _BP_METRIC_SCALED else raw
                )

        feat_results: dict[str, 'dict | None'] = {}
        for fn in feat_names:
            bp = feat_bp[fn]
            if np.isnan(bp.get('band_depth', float('nan'))):
                feat_results[fn] = None
            else:
                for key in _BP_METRIC_KEYS:
                    bp.setdefault(key, float('nan'))
                feat_results[fn] = bp
        results[sp_name] = feat_results

    return {'features': features, 'results': results, 'sources': sources}
