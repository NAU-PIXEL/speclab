#!/usr/bin/env python3
"""
Test plot_instrument_metrics() with example_data/WardRocks_igneous1.

Three cases:
  1. From CSV filepath — all channels (default).
  2. From CSV filepath — temperature only (no BB resistance).
  3. From emcal output notes dict — all channels.
"""

import os
import logging

from speclab import emcal
from speclab.plot import plot_instrument_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

_HERE  = os.path.dirname(os.path.abspath(__file__))
FDIR   = os.path.join(_HERE, '..', 'example_data', 'WardRocks_igneous1')
CSV    = os.path.join(FDIR, 'measurement-info.csv')

# ---------------------------------------------------------------------------
# Case 1 — filepath, all channels
# ---------------------------------------------------------------------------
print("\n--- Case 1: filepath, all channels ---")
plot_instrument_metrics(CSV)

# ---------------------------------------------------------------------------
# Case 2 — filepath, temperature only
# ---------------------------------------------------------------------------
print("\n--- Case 2: filepath, temperature only (no BB resistance) ---")
plot_instrument_metrics(CSV, temp_channels=[103, 104, 105, 106, 107], bb_channels=[])

# ---------------------------------------------------------------------------
# Case 3 — from emcal notes dict
# ---------------------------------------------------------------------------
print("\n--- Case 3: emcal notes dict ---")
out = emcal(fdir=FDIR, lab='nau', method='nem', save=False, plot=False)
plot_instrument_metrics(out['notes'], out['label'])
