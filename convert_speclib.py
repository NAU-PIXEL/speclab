#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_speclib.py – Convert any DV-format spectral-library HDF5 to the
SpectralViewer per-spectrum format.

Three on-disk DV layouts are handled:

    flat DaVinci  — top-level datasets: ``data`` (n_spec, n_pts),
                    ``xaxis`` (n_pts,), string fields packed as
                    newline-delimited byte strings in shape (1,).
                    This layout breaks ``sma`` because ``readDVhdf``
                    returns a field-name dict rather than an album dict.

    per-spectrum  — one HDF5 group per spectrum keyed by spec-id or
                    sequential integer.  Already readable by ``sma``;
                    conversion re-keys to sequential 0-based integers
                    and normalises dtypes.

    grouped SV    — groups named ``group001``, ``group002``, …, each
                    holding a shared ``xaxis`` and a 2-D ``data``
                    matrix.  Expanded into per-spectrum entries.

Output (SpectralViewer per-spectrum format):

    One HDF5 group per spectrum, keyed ``'0'``, ``'1'``, …
    Each group contains:
        data   : float64 (n_pts,)
        xaxis  : float64 (n_pts,)
        <metadata fields> : 0-d datasets (numeric) or byte strings
"""

import logging
import os

from .utils import readDVhdf, saveDVhdf, dv_to_album

log = logging.getLogger(__name__)


def convert_dv_to_sv(input_path: str, output_path: str) -> dict:
    """
    Load a DV-format spectral library and save it as a SpectralViewer
    per-spectrum HDF5 file.

    Parameters
    ----------
    input_path : str
        Path to the source HDF5 file (any DV layout).
    output_path : str
        Destination path for the converted file.

    Returns
    -------
    dict
        The album dict that was written (sequential integer keys).
    """
    log.info("Loading '%s'", input_path)
    raw   = readDVhdf(input_path)
    album = dv_to_album(raw)
    log.info("Converted %d spectra; saving to '%s'", len(album), output_path)

    sv_dict = {
        str(sid): {k: v for k, v in entry.items()}
        for sid, entry in album.items()
    }
    saveDVhdf(sv_dict, output_path)
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
    2. Output file is readable by ``readDVhdf``.
    3. Every top-level value in the output is a sub-dict (album format).
    4. Every entry has 1-D ``data`` and ``xaxis`` of matching length.
    5. Spectral data values are identical (within float32 precision) to
       those loaded via the original file through ``dv_to_album``.
    6. Output file can be loaded as an ``sma``-compatible endlib
       (``dv_to_album`` succeeds on the round-tripped file).

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
        reloaded_raw = readDVhdf(output_path)
        print("[2] Output readable by readDVhdf ✓")

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
        rt_album = dv_to_album(reloaded_raw)
        assert len(rt_album) == n, (
            f"Round-trip album has {len(rt_album)} entries, expected {n}"
        )
        first = next(iter(rt_album.values()))
        assert 'data'  in first and 'xaxis' in first
        print("[6] Round-trip output passes dv_to_album (sma-compatible) ✓")

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
        description='Convert a DV-format speclib HDF5 to SpectralViewer format.'
    )
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_conv = sub.add_parser('convert', help='Convert a file')
    p_conv.add_argument('input',  help='Source HDF5 file')
    p_conv.add_argument('output', help='Destination HDF5 file')

    p_test = sub.add_parser('test', help='Test conversion of a file')
    p_test.add_argument('input',  help='Source HDF5 file')
    p_test.add_argument('output', nargs='?', default=None,
                        help='Destination HDF5 file (default: temp file)')

    args = parser.parse_args()

    if args.cmd == 'convert':
        convert_dv_to_sv(args.input, args.output)
    elif args.cmd == 'test':
        test_conversion(args.input, args.output)
