#!/usr/bin/env python3
"""
Generate dummy VSWIR reflectance CSV files for GUI testing.

Outputs
-------
example_data/dummy_vswir/data.csv
    Wavelength 390–2500 nm at 1 nm spacing.
    12 columns: sample_01 … sample_12.

example_data/dummy_vswir/library.csv
    Wavelength 400–2300 nm at 2 nm spacing (different grid → tests resampling).
    12 columns: reference_01 … reference_12.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_DIR = Path(__file__).resolve().parent.parent / 'example_data' / 'dummy_vswir'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Wavelength grids
# ---------------------------------------------------------------------------

WL_DATA = np.arange(390, 2501, 1, dtype=float)    # 2111 points, 1 nm
WL_LIB  = np.arange(400, 2301, 2, dtype=float)    #  951 points, 2 nm

# ---------------------------------------------------------------------------
# Spectral shape parameters
# ---------------------------------------------------------------------------

# (centre_nm, typical_depth, half-width_nm) for common VSWIR absorption features.
_FEATURES = [
    ( 680, 0.08,  30),   # chlorophyll red absorption
    ( 970, 0.04,  35),   # water overtone
    (1200, 0.06,  45),   # water combination band
    (1400, 0.22,  60),   # strong water absorption
    (1900, 0.26,  70),   # strong water / CO2 absorption
    (2200, 0.12,  50),   # Al-OH / carbonate feature
    (2340, 0.09,  40),   # carbonate / C-H overtone
]


def _spectrum(wl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one plausible reflectance curve on *wl*.

    The base shape is a smooth sinusoidal envelope; absorption features are
    Gaussian dips with per-spectrum depth jitter.
    """
    # Smooth background: linear slope + two sinusoidal undulations
    slope = rng.uniform(-0.05, 0.05)
    base  = (rng.uniform(0.15, 0.55)
             + slope * (wl - wl.mean()) / (wl[-1] - wl[0])
             + 0.06 * np.sin((wl - 400) / 2100 * np.pi * 2.4)
             + 0.03 * np.cos((wl - 400) / 2100 * np.pi * 5.1))

    for centre, depth, hw in _FEATURES:
        d = depth * rng.uniform(0.4, 1.6)
        base -= d * np.exp(-((wl - centre) ** 2) / (2 * hw ** 2))

    base += rng.normal(0, 0.001, len(wl))
    return np.clip(base, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Generate and save
# ---------------------------------------------------------------------------

def _make_csv(wl: np.ndarray, col_prefix: str, n: int, seed: int, path: Path) -> None:
    rng  = np.random.default_rng(seed)
    cols = {f'{col_prefix}_{i + 1:02d}': _spectrum(wl, rng) for i in range(n)}
    df   = pd.DataFrame({'Wavelength': wl, **cols})
    df.to_csv(path, index=False, float_format='%.6f')
    print(f'✓  {path.relative_to(path.parent.parent.parent)}  '
          f'({len(wl)} pts × {n} spectra)')


_make_csv(WL_DATA, 'sample',    12, seed=42,  path=OUT_DIR / 'data.csv')
_make_csv(WL_LIB,  'reference', 12, seed=137, path=OUT_DIR / 'library.csv')