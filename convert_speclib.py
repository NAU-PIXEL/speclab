#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_speclib.py – Spectral library conversion utilities.

DV-format HDF5 conversion (convert_dv_to_sv)
---------------------------------------------
Three on-disk DV layouts are handled:

    flat DaVinci  — top-level datasets: ``data`` (n_spec, n_pts),
                    ``xaxis`` (n_pts,), string fields packed as
                    newline-delimited byte strings in shape (1,).
                    This layout breaks ``sma`` because ``readHDF``
                    returns a field-name dict rather than an album dict.

    per-spectrum  — one HDF5 group per spectrum keyed by spec-id or
                    sequential integer.  Already readable by ``sma``;
                    conversion re-keys to sequential 0-based integers
                    and normalises dtypes.

    grouped SV    — groups named ``group001``, ``group002``, …, each
                    holding a shared ``xaxis`` and a 2-D ``data``
                    matrix.  Expanded into per-spectrum entries.

USGS Spectral Library Version 7 conversion (convert_usgs_splib07)
------------------------------------------------------------------
Reads the ASCII text files from the splib07 download and writes a
single speclab-compatible HDF5 file.  Supported source:

    cvASD — all spectra convolved to ASD standard-resolution grid
            (0.35–2.5 µm, 2151 channels); wavelength axis stored in nm.

Output (per-spectrum format):

    One HDF5 group per spectrum, keyed ``'0'``, ``'1'``, …
    Each group contains:
        data              : float64 (n_pts,)  — reflectance, NaN where bad
        xaxis             : float64 (n_pts,)  — wavelength in nm
        label / sample_name : str             — display label
        mineral_name      : str
        sample_id         : str
        spectrometer      : str
        meas_type         : str  (AREF | RREF | RTGC)
        chapter           : str  (Minerals | Vegetation | …)
        mineral_type      : str  — from HTML metadata, when available
        formula           : str  — chemical formula, when available
        collection_locality : str
        original_donor    : str
        source_library    : str  ("USGS splib07b_cvASD")

CRISM Spectral Library conversion (convert_crism_speclib)
---------------------------------------------------------
Reads a DaVinci-format CRISM HDF5 and writes a speclab-compatible
per-spectrum HDF5.  The source file uses a flat layout with:

    * Metadata packed as newline-delimited byte strings (parallel arrays).
    * Spectral data stored per-specimen as ``(1, n_pts, n_vars)`` arrays
      where column 0 = wavelength (µm) and column 1 = reflectance.

Output (per-spectrum format):

    One HDF5 group per spectrum, keyed ``'0'``, ``'1'``, …
    Each group contains:
        data              : float64 (n_pts,)  — reflectance (column 1 of source)
        xaxis             : float64 (n_pts,)  — wavelength in nm (µm × 1000)
        label / sample_name : str             — "{specimen_name} [{type}]"
        specimen_id       : str               — lab sample ID (source 'name' field)
        specimen_name     : str               — specimen name
        crism_id          : str               — CRISM internal ID
        type              : str               — VNIR | VNIR-MIR | NIR-MIR | VNIR-FIR
        body              : str               — EARTH | MOON | MARS | METEORITE | …
        material          : str               — MINERAL | ROCK | INORGANIC | …
        mineral_family    : str               — CARBONATE | SULFATE | IGNEOUS | …
        mineral_name      : str               — specific mineral / rock subtype
        current_location  : str               — repository holding the sample
        collection_location : str             — origin locality
        reference         : str               — citation key
        source_library    : str               — "CRISM Spectral Library"
"""

import logging
import os
import re
from html import unescape
from pathlib import Path

import numpy as np

from .utils import readHDF, saveHDF, _to_album

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# USGS splib07 constants
# ---------------------------------------------------------------------------

_USGS_BAD_VALUE: float = -1.23e34

_USGS_CHAPTER_NAMES: dict[str, str] = {
    'ChapterA_ArtificialMaterials': 'Artificial Materials',
    'ChapterC_Coatings':            'Coatings',
    'ChapterL_Liquids':             'Liquids',
    'ChapterM_Minerals':            'Minerals',
    'ChapterO_OrganicCompounds':    'Organic Compounds',
    'ChapterS_SoilsAndMixtures':    'Soils & Mixtures',
    'ChapterV_Vegetation':          'Vegetation',
}

# HTML metadata keys → HDF field names
_USGS_HTML_KEYS: dict[str, str] = {
    'MINERAL_TYPE':        'mineral_type',
    'MINERAL':             'mineral',
    'FORMULA':             'formula',
    'COLLECTION_LOCALITY': 'collection_locality',
    'ORIGINAL_DONOR':      'original_donor',
}

# Regex: matches a USGS metadata key followed by colon (not FORMULA_HTML, etc.)
_USGS_HTML_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _USGS_HTML_KEYS) + r'):\s*'
    r'(.*?)(?=\s+[A-Z][A-Z_]*:|\s*$)',
)


# ---------------------------------------------------------------------------
# USGS splib07 helpers
# ---------------------------------------------------------------------------

def _usgs_load_wavelengths(wl_file: Path) -> np.ndarray:
    """
    Parse a USGS ASCII wavelength file and return the axis in nm.

    Parameters
    ----------
    wl_file : Path
        Path to a ``*_Wavelengths_*.txt`` file (header line + one float per line).

    Returns
    -------
    np.ndarray
        Wavelength axis in nm, shape (n_channels,).
    """
    lines = wl_file.read_text(encoding='utf-8', errors='replace').splitlines()
    wl_um = np.array([float(ln) for ln in lines[1:] if ln.strip()])
    return wl_um * 1000.0   # µm → nm


def _usgs_load_spectrum(txt_file: Path) -> tuple[str, np.ndarray]:
    """
    Load a single USGS ASCII spectrum file.

    Parameters
    ----------
    txt_file : Path
        Path to a spectrum ``.txt`` file.

    Returns
    -------
    title : str
        The portion of the header line after ``Record=NNN:``, whitespace-normalised.
    data : np.ndarray
        Reflectance values with bad-band sentinel (-1.23e34) replaced by NaN.
    """
    lines = txt_file.read_text(encoding='utf-8', errors='replace').splitlines()
    header = lines[0].strip()
    title  = header.split(':', 1)[1].strip() if ':' in header else header
    # Normalise internal runs of whitespace to a single space for consistent parsing
    title  = re.sub(r'\s+', ' ', title)
    values = np.array([float(ln) for ln in lines[1:] if ln.strip()])
    values[values < _USGS_BAD_VALUE * 0.5] = np.nan   # mask sentinel
    return title, values


def _usgs_parse_title(title: str) -> dict[str, str]:
    """
    Decompose a whitespace-normalised USGS spectrum title into components.

    The title (after stripping the ``Record=NNN:`` prefix) has the form::

        {name tokens...} {spectrometer} {meas_type}

    where the last two whitespace-separated tokens are always the spectrometer
    code and the measurement type (AREF / RREF / RTGC).

    Parameters
    ----------
    title : str
        Whitespace-normalised title string, e.g.
        ``"Actinolite HS116.1B ASDFRb AREF"``.

    Returns
    -------
    dict
        Keys: ``mineral_name``, ``sample_id``, ``spectrometer``,
        ``meas_type``, ``label``, ``sample_name``.
    """
    tokens = title.split()
    if len(tokens) >= 2:
        meas_type    = tokens[-1]
        spectrometer = tokens[-2]
        name_tokens  = tokens[:-2]
    elif len(tokens) == 1:
        meas_type = spectrometer = ''
        name_tokens = tokens
    else:
        return {k: '' for k in ('mineral_name', 'sample_id', 'spectrometer',
                                 'meas_type', 'label', 'sample_name')}

    mineral_name = name_tokens[0] if name_tokens else ''
    sample_id    = ' '.join(name_tokens[1:]) if len(name_tokens) > 1 else ''
    label        = f"{' '.join(name_tokens)} [{spectrometer}]"

    return {
        'mineral_name': mineral_name,
        'sample_id':    sample_id,
        'spectrometer': spectrometer,
        'meas_type':    meas_type,
        'label':        label,
        'sample_name':  label,
    }


def _usgs_parse_html(html_path: Path) -> dict[str, str]:
    """
    Extract structured metadata from a USGS HTML sample-description file.

    Parameters
    ----------
    html_path : Path
        Path to the ``.html`` metadata file.  Missing files are silently ignored.

    Returns
    -------
    dict
        A subset of ``_USGS_HTML_KEYS`` values populated from the file.
    """
    if not html_path.is_file():
        return {}
    try:
        raw = html_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return {}

    # Strip HTML tags, decode entities, collapse whitespace to single spaces
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = unescape(text)
    text = re.sub(r'\s+', ' ', text)

    result: dict[str, str] = {}
    for m in _USGS_HTML_PATTERN.finditer(text):
        html_key = m.group(1)
        val      = m.group(2).strip()
        field    = _USGS_HTML_KEYS[html_key]
        if val and not val.startswith('END_') and field not in result:
            result[field] = val
    return result


# ---------------------------------------------------------------------------
# USGS splib07 main conversion
# ---------------------------------------------------------------------------

def convert_usgs_splib07(
    splib_dir: 'str | Path',
    output_path: 'str | Path',
    *,
    source: str = 'cvASD',
    chapters: 'list[str] | None' = None,
    drop_all_nan: bool = True,
) -> int:
    """
    Convert the USGS Spectral Library Version 7 (splib07) to speclab HDF format.

    Reads the ASCII text files from the splib07 download directory and writes
    a single speclab-compatible per-spectrum HDF5 file, readable by
    ``readHDF`` / ``_to_album``.

    Parameters
    ----------
    splib_dir : str or Path
        Top-level directory of the splib07 download (contains ``ASCIIdata/``
        and ``HTMLmetadata/``).
    output_path : str or Path
        Destination HDF5 file path.
    source : str
        Which version of the library to read.  Currently only ``'cvASD'``
        is supported: all spectra convolved to the ASD standard-resolution
        grid (0.35–2.5 µm, 2151 channels).  The wavelength axis is stored
        in nm to match ``loadASD`` conventions.
    chapters : list of str or None
        Subset of chapter names to include, e.g. ``['Minerals', 'Vegetation']``.
        Accepts either the full folder name (``'ChapterM_Minerals'``) or the
        human-readable label (``'Minerals'``).  ``None`` includes all chapters.
    drop_all_nan : bool
        If True (default), skip spectra whose entire reflectance array is NaN
        after masking bad-band sentinels.  These arise when the original
        measurement was made on an instrument with no overlap with the ASD
        range (e.g. Nicolet FTIR beyond 2.5 µm).

    Returns
    -------
    int
        Number of spectra written to the output file.
    """
    import h5py

    if source != 'cvASD':
        raise NotImplementedError(
            f"source={source!r} is not yet supported; only 'cvASD' is implemented."
        )

    splib_dir   = Path(splib_dir)
    output_path = Path(output_path)

    ascii_dir   = splib_dir / 'ASCIIdata' / 'ASCIIdata_splib07b_cvASD'
    html_dir    = splib_dir / 'HTMLmetadata'
    file_prefix = 's07_ASD_'
    wl_filename = 's07_ASD_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt'

    if not ascii_dir.is_dir():
        raise FileNotFoundError(f"Expected ASCIIdata directory not found: {ascii_dir}")

    # Shared wavelength axis (µm → nm)
    wl_file = ascii_dir / wl_filename
    xaxis   = _usgs_load_wavelengths(wl_file)
    n_ch    = len(xaxis)
    log.info("Wavelength axis: %d channels, %.1f–%.1f nm", n_ch, xaxis[0], xaxis[-1])

    # Resolve chapters
    all_chapters: dict[str, str] = {
        d.name: _USGS_CHAPTER_NAMES.get(d.name, d.name.split('_', 1)[-1])
        for d in sorted(ascii_dir.iterdir())
        if d.is_dir() and d.name.startswith('Chapter')
    }
    if chapters is not None:
        all_chapters = {
            k: v for k, v in all_chapters.items()
            if v in chapters or k in chapters
        }
    if not all_chapters:
        raise ValueError(f"No matching chapters found for filter: {chapters}")

    log.info("Chapters: %s", ', '.join(all_chapters.values()))

    n_written = 0
    n_skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as hf:
        hf.attrs['source_library'] = 'USGS splib07b_cvASD'
        hf.attrs['n_channels']     = n_ch

        for ch_dir_name, ch_label in all_chapters.items():
            ch_dir    = ascii_dir / ch_dir_name
            txt_files = sorted(
                f for f in ch_dir.iterdir()
                if f.suffix == '.txt'
                   and not f.name.startswith('s07_ASD_Wavelengths')
                   and not f.name.startswith('s07_ASD_Bandpass')
            )
            log.info("  %s: %d spectra", ch_label, len(txt_files))

            for txt_path in txt_files:
                # --- load spectrum ------------------------------------------
                title, data = _usgs_load_spectrum(txt_path)

                if len(data) != n_ch:
                    log.warning("    skip %s: %d channels ≠ %d",
                                txt_path.name, len(data), n_ch)
                    n_skipped += 1
                    continue

                if drop_all_nan and np.all(np.isnan(data)):
                    n_skipped += 1
                    continue

                # --- metadata -----------------------------------------------
                stem     = txt_path.stem[len(file_prefix):]   # strip s07_ASD_
                html_path = html_dir / (stem + '.html')

                meta = _usgs_parse_title(title)
                meta.update(_usgs_parse_html(html_path))
                meta['chapter']        = ch_label
                meta['source_library'] = 'USGS splib07b_cvASD'
                meta['sample_name']    = meta['label']
                meta['x_unit']         = 'nm'
                meta['y_unit']         = 'reflectance'

                # --- write HDF5 group ---------------------------------------
                grp = hf.create_group(str(n_written))
                grp.create_dataset(
                    'data', data=data, dtype=np.float64,
                    compression='gzip', compression_opts=4, shuffle=True,
                )
                grp.create_dataset(
                    'xaxis', data=xaxis, dtype=np.float64,
                    compression='gzip', compression_opts=4, shuffle=True,
                )
                for field, val in meta.items():
                    if val:
                        grp.create_dataset(field, data=val.encode('utf-8'))

                n_written += 1

    log.info("✓ Wrote %d spectra (%d skipped all-NaN or wrong shape) → '%s'",
             n_written, n_skipped, output_path)
    return n_written


# ---------------------------------------------------------------------------
# CRISM spectral library constants
# ---------------------------------------------------------------------------

# Indices into the comma-separated 8-level class hierarchy
_CRISM_CLASS_BODY   = 2   # EARTH | MOON | MARS | METEORITE | MIXTURE | …
_CRISM_CLASS_MAT    = 3   # MINERAL | ROCK | INORGANIC | UNCONSOLIDATED
_CRISM_CLASS_FAM    = 5   # CARBONATE | SULFATE | IGNEOUS | PHYLLOSILICATE | …
_CRISM_CLASS_NAME   = 7   # specific mineral/rock name

_CRISM_UNCLASSIFIED = frozenset({'UNCLASSIFIED', ''})


# ---------------------------------------------------------------------------
# CRISM spectral library helpers
# ---------------------------------------------------------------------------

def _crism_strip(val: 'str | bytes | np.str_') -> str:
    """Strip surrounding double-quotes and whitespace from a CRISM string value."""
    return str(val).strip().strip('"')


def _crism_parse_class(class_str: str) -> dict[str, str]:
    """
    Decompose the CRISM 8-level comma-separated class hierarchy.

    Parameters
    ----------
    class_str : str
        Raw class string, e.g.
        ``"NATURAL, SOLID, EARTH, MINERAL, VOLATILE-POOR, CARBONATE, UNCLASSIFIED, CALCITE"``.

    Returns
    -------
    dict
        Keys: ``body``, ``material``, ``mineral_family``, ``mineral_name``.
        Values are the level text, or empty string for UNCLASSIFIED/missing levels.
    """
    parts = [p.strip() for p in class_str.split(',')]
    # Pad if the class string has fewer than 8 levels
    while len(parts) < 8:
        parts.append('UNCLASSIFIED')

    def _clean(s: str) -> str:
        return '' if s.upper() in _CRISM_UNCLASSIFIED else s

    return {
        'body':           _clean(parts[_CRISM_CLASS_BODY]),
        'material':       _clean(parts[_CRISM_CLASS_MAT]),
        'mineral_family': _clean(parts[_CRISM_CLASS_FAM]),
        'mineral_name':   _clean(parts[_CRISM_CLASS_NAME]),
    }


# ---------------------------------------------------------------------------
# CRISM spectral library main conversion
# ---------------------------------------------------------------------------

def convert_crism_speclib(
    input_path: 'str | Path',
    output_path: 'str | Path',
    *,
    drop_all_nan: bool = True,
) -> int:
    """
    Convert a DaVinci-format CRISM Spectral Library HDF5 to speclab HDF format.

    The source file uses a flat DaVinci layout: metadata fields are packed as
    newline-delimited byte strings and spectral data are stored per specimen as
    ``(1, n_pts, n_vars)`` arrays (column 0 = wavelength in µm, column 1 =
    reflectance).  The BSQ reversal applied by ``readHDF`` mangles this
    multi-variable layout, so spectral data are read directly via h5py.

    Parameters
    ----------
    input_path : str or Path
        Path to the source CRISM HDF5 file.
    output_path : str or Path
        Destination HDF5 file path.
    drop_all_nan : bool
        If True (default), skip spectra whose entire reflectance array is NaN.

    Returns
    -------
    int
        Number of spectra written to the output file.
    """
    import h5py

    input_path  = Path(input_path)
    output_path = Path(output_path)

    if not input_path.is_file():
        raise FileNotFoundError(f"CRISM HDF5 not found: {input_path}")

    # ---- load metadata via readHDF (handles packed byte strings) ----------
    meta_raw = readHDF(str(input_path), collapse=False)

    def _meta_array(key: str) -> np.ndarray:
        """Return the metadata string array for key, or an empty array."""
        val = meta_raw.get(key, np.array([]))
        if isinstance(val, np.ndarray):
            return val
        return np.array([str(val)])

    ids_arr       = _meta_array('id')
    names_arr     = _meta_array('name')
    spec_names_arr = _meta_array('specimen_name')
    types_arr     = _meta_array('type')
    classes_arr   = _meta_array('class')
    refs_arr      = _meta_array('reference')
    cur_loc_arr   = _meta_array('current_location')
    col_loc_arr   = _meta_array('collection_location')

    n_meta = len(ids_arr)
    log.info("CRISM: %d metadata entries from '%s'", n_meta, input_path.name)

    # ---- open source and destination HDF5 files -----------------------------
    n_written = 0
    n_skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:
        dst.attrs['source_library'] = 'CRISM Spectral Library'

        data_grp = src['data']

        for i in range(n_meta):
            data_key = f'id{i + 1:04d}'

            if data_key not in data_grp:
                log.warning("  missing data key %s (index %d) — skip", data_key, i)
                n_skipped += 1
                continue

            # Shape: (1, n_pts, n_vars); col 0 = µm, col 1 = reflectance
            raw_arr = data_grp[data_key][()]  # (1, n_pts, n_vars)
            if raw_arr.ndim != 3 or raw_arr.shape[0] != 1 or raw_arr.shape[2] < 2:
                log.warning("  unexpected shape %s for %s — skip", raw_arr.shape, data_key)
                n_skipped += 1
                continue

            xaxis = raw_arr[0, :, 0].astype(np.float64) * 1000.0  # µm → nm
            data  = raw_arr[0, :, 1].astype(np.float64)

            if drop_all_nan and np.all(np.isnan(data)):
                n_skipped += 1
                continue

            # ---- parse metadata --------------------------------------------
            crism_id      = _crism_strip(ids_arr[i])
            specimen_id   = _crism_strip(names_arr[i])
            specimen_name = _crism_strip(spec_names_arr[i])
            spec_type     = _crism_strip(types_arr[i])
            class_str     = _crism_strip(classes_arr[i])
            reference     = _crism_strip(refs_arr[i])      if i < len(refs_arr)    else ''
            cur_loc       = _crism_strip(cur_loc_arr[i])   if i < len(cur_loc_arr) else ''
            col_loc       = _crism_strip(col_loc_arr[i])   if i < len(col_loc_arr) else ''

            class_fields = _crism_parse_class(class_str)

            display_name = specimen_name or specimen_id or crism_id
            label        = f"{display_name} [{spec_type}]" if spec_type else display_name

            # ---- write HDF5 group ------------------------------------------
            grp = dst.create_group(str(n_written))
            grp.create_dataset(
                'data',  data=data,  dtype=np.float64,
                compression='gzip', compression_opts=4, shuffle=True,
            )
            grp.create_dataset(
                'xaxis', data=xaxis, dtype=np.float64,
                compression='gzip', compression_opts=4, shuffle=True,
            )

            str_fields = {
                'label':               label,
                'sample_name':         label,
                'specimen_name':       specimen_name,
                'specimen_id':         specimen_id,
                'crism_id':            crism_id,
                'spectral_range':      spec_type,
                'reference':           reference,
                'current_location':    cur_loc,
                'collection_location': col_loc,
                'source_library':      'CRISM Spectral Library',
                'x_unit':              'nm',
                'y_unit':              'reflectance',
                **class_fields,
            }
            for field, val in str_fields.items():
                if val:
                    grp.create_dataset(field, data=val.encode('utf-8'))

            n_written += 1

    log.info("✓ Wrote %d spectra (%d skipped) → '%s'", n_written, n_skipped, output_path)
    return n_written


# ---------------------------------------------------------------------------
# DV → SpectralViewer conversion
# ---------------------------------------------------------------------------

def convert_dv_to_sv(
    input_path: str,
    output_path: str,
    *,
    x_unit: str = 'cm-1',
    y_unit: str = 'emissivity',
) -> dict:
    """
    Load a DV-format spectral library and save it as a SpectralViewer
    per-spectrum HDF5 file.

    Parameters
    ----------
    input_path : str
        Path to the source HDF5 file (any DV layout).
    output_path : str
        Destination path for the converted file.
    x_unit : str
        X-axis unit tag written to every spectrum.  Use ``'cm-1'`` for
        wavenumber (TIR, default), ``'um'`` for wavelength in µm (TIR),
        or ``'nm'`` for wavelength in nm (VNIR).
    y_unit : str
        Y-axis unit tag written to every spectrum.  Use ``'emissivity'``
        (default), ``'reflectance'``, or ``'transmittance'``.

    Returns
    -------
    dict
        The album dict that was written (sequential integer keys).
    """
    log.info("Loading '%s'", input_path)
    raw   = readHDF(input_path)
    album = _to_album(raw)
    log.info("Converted %d spectra; saving to '%s'", len(album), output_path)

    sv_dict = {
        str(sid): {**entry, 'x_unit': x_unit, 'y_unit': y_unit}
        for sid, entry in album.items()
    }
    saveHDF(sv_dict, output_path)
    log.info("✓ Saved %s", output_path)
    return album


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------

def test_conversion(input_path: str, output_path: str | None = None) -> None:
    """
    Verify that *input_path* converts cleanly to the SV per-spectrum
    format and that the output passes structural checks needed by ``sma``.

    Checks performed
    ----------------
    1. Conversion runs without error.
    2. Output file is readable by ``readHDF``.
    3. Every top-level value in the output is a sub-dict (album format).
    4. Every entry has 1-D ``data`` and ``xaxis`` of matching length.
    5. Spectral data values are identical (within float32 precision) to
       those loaded via the original file through ``_to_album``.
    6. Output file can be loaded as an ``sma``-compatible endlib
       (``_to_album`` succeeds on the round-tripped file).

    Parameters
    ----------
    input_path : str
        Path to the source DV-format HDF5 file.
    output_path : str or None
        Path for the converted output.  If None a temporary path in
        ``/tmp`` is used and deleted after the test.
    """
    import tempfile

    tmp = output_path is None
    if tmp:
        fd, output_path = tempfile.mkstemp(suffix='.hdf', prefix='sv_test_')
        os.close(fd)

    try:
        # ---- Step 1: Convert -----------------------------------------------
        album = convert_dv_to_sv(input_path, output_path)
        n = len(album)
        assert n > 0, "Conversion produced an empty album"
        print(f"[1] Conversion OK — {n} spectra")

        # ---- Step 2: Reload -------------------------------------------------
        reloaded_raw = readHDF(output_path)
        print("[2] Output readable by readHDF ✓")

        # ---- Step 3: Album format -------------------------------------------
        non_dict = [k for k, v in reloaded_raw.items() if not isinstance(v, dict)]
        assert not non_dict, f"Non-dict top-level keys after reload: {non_dict}"
        print("[3] All top-level values are sub-dicts (album format) ✓")

        # ---- Step 4: 1-D data / xaxis in every entry ------------------------
        for sid, entry in reloaded_raw.items():
            assert 'data'  in entry, f"Entry {sid} missing 'data'"
            assert 'xaxis' in entry, f"Entry {sid} missing 'xaxis'"
            d = np.asarray(entry['data'])
            x = np.asarray(entry['xaxis'])
            assert d.ndim == 1, f"Entry {sid}: data.ndim={d.ndim}, expected 1"
            assert x.ndim == 1, f"Entry {sid}: xaxis.ndim={x.ndim}, expected 1"
            assert d.shape == x.shape, (
                f"Entry {sid}: data.shape={d.shape} ≠ xaxis.shape={x.shape}"
            )
        print("[4] All entries have matching 1-D data and xaxis ✓")

        # ---- Step 5: Data round-trip fidelity --------------------------------
        for i, (sid_orig, entry_orig) in enumerate(album.items()):
            sid_str = str(sid_orig)
            entry_rt = reloaded_raw[sid_str]
            orig_d = np.asarray(entry_orig['data'],  dtype=np.float64)
            rt_d   = np.asarray(entry_rt['data'],    dtype=np.float64)
            nan_match = np.array_equal(np.isnan(orig_d), np.isnan(rt_d))
            assert nan_match, (
                f"Spectrum {sid_orig}: NaN positions differ after round-trip"
            )
            mask = ~np.isnan(orig_d)
            assert np.allclose(orig_d[mask], rt_d[mask], rtol=1e-5, atol=1e-7), (
                f"Spectrum {sid_orig}: round-trip data mismatch "
                f"(max_diff={np.abs(orig_d[mask] - rt_d[mask]).max():.3e})"
            )
        print("[5] Spectral data round-trip fidelity ✓")

        # ---- Step 6: sma-compatible endlib ----------------------------------
        rt_album = _to_album(reloaded_raw)
        assert len(rt_album) == n, (
            f"Round-trip album has {len(rt_album)} entries, expected {n}"
        )
        first = next(iter(rt_album.values()))
        assert 'data'  in first and 'xaxis' in first
        print("[6] Round-trip output passes _to_album (sma-compatible) ✓")

        print(f"\nAll checks passed for '{input_path}'")

    finally:
        if tmp and os.path.exists(output_path):
            os.unlink(output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(
        description='Spectral library conversion utilities.'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_conv = sub.add_parser('convert', help='Convert a DV-format HDF5 to SpectralViewer format')
    p_conv.add_argument('input',  help='Source HDF5 file')
    p_conv.add_argument('output', help='Destination HDF5 file')

    p_test = sub.add_parser('test', help='Test conversion of a DV-format HDF5 file')
    p_test.add_argument('input',  help='Source HDF5 file')
    p_test.add_argument('output', nargs='?', default=None,
                        help='Destination HDF5 file (default: temp file)')

    p_usgs = sub.add_parser('usgs', help='Convert USGS splib07 to speclab HDF format')
    p_usgs.add_argument('splib_dir', help='Top-level usgs_splib07 directory')
    p_usgs.add_argument('output',    help='Destination HDF5 file')
    p_usgs.add_argument(
        '--source', default='cvASD',
        help="Library variant to read (default: cvASD)",
    )
    p_usgs.add_argument(
        '--chapters', nargs='+', default=None, metavar='CHAPTER',
        help=(
            "Chapters to include, e.g. Minerals Vegetation "
            "(default: all chapters)"
        ),
    )
    p_usgs.add_argument(
        '--keep-all-nan', action='store_true',
        help="Keep spectra that are entirely NaN after bad-band masking",
    )

    p_crism = sub.add_parser('crism', help='Convert a CRISM Spectral Library HDF5 to speclab format')
    p_crism.add_argument('input',  help='Source CRISM HDF5 file')
    p_crism.add_argument('output', help='Destination HDF5 file')
    p_crism.add_argument(
        '--keep-all-nan', action='store_true',
        help="Keep spectra whose reflectance array is entirely NaN",
    )

    args = parser.parse_args()

    if args.cmd == 'convert':
        convert_dv_to_sv(args.input, args.output)
    elif args.cmd == 'test':
        test_conversion(args.input, args.output)
    elif args.cmd == 'usgs':
        n = convert_usgs_splib07(
            args.splib_dir,
            args.output,
            source=args.source,
            chapters=args.chapters,
            drop_all_nan=not args.keep_all_nan,
        )
        print(f"Done: {n} spectra written to '{args.output}'")
    elif args.cmd == 'crism':
        n = convert_crism_speclib(
            args.input,
            args.output,
            drop_all_nan=not args.keep_all_nan,
        )
        print(f"Done: {n} spectra written to '{args.output}'")
