#!/usr/bin/env python3
"""
compare_wards_spectra.py

Compare measured emissivity (emcal_merged_results.hdf) against the
CSE Wards spectral library (CSE_wards_rock_library_SV.hdf) for all
samples with matching names.

Navigation
----------
  ← / →   previous / next matched pair
  q        quit
"""

import os
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from speclab.utils import readDVhdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

CSE_PATH = ROOT / 'spectral_libraries' / 'CSE_wards_rock_library_SV.hdf'

_nc = next(
    d for d in os.listdir(os.path.expanduser('~/Library/CloudStorage'))
    if 'Nextcloud' in d
)
EMCAL_PATH = Path(os.path.expanduser('~/Library/CloudStorage')) / _nc / \
    '107 Storage/FTIR Data/jfsmekens/WardRocks/emcal_merged_results.hdf'

# ---------------------------------------------------------------------------
# Load CSE library
# ---------------------------------------------------------------------------

print(f'Loading CSE library: {CSE_PATH.name}')
cse = readDVhdf(str(CSE_PATH))
cse_names   = [str(n).strip() for n in cse['sample_name']]
cse_data    = cse['data']       # (100, 936)
cse_xaxis   = cse['xaxis']     # (936,)

# ---------------------------------------------------------------------------
# Load emcal results
# ---------------------------------------------------------------------------

print(f'Loading emcal results: {EMCAL_PATH.name}')
with h5py.File(EMCAL_PATH, 'r') as f:
    em_xaxis  = f['xaxis'][()]                           # (920,)
    em_labels = list(f['emiss'].keys())                  # '001 - Biotite Granite', ...
    em_data   = {k: f['emiss'][k][()] for k in em_labels}

# ---------------------------------------------------------------------------
# Match on rock name (strip leading 'NNN - ' prefix from emcal labels)
# ---------------------------------------------------------------------------

matches: list[tuple[str, str, int]] = []   # (emcal_label, cse_name, cse_idx)

for em_lbl in em_labels:
    rock_name = em_lbl.split(' - ', 1)[-1].strip()
    # exact match first, then case-insensitive
    for ci, cn in enumerate(cse_names):
        if cn.lower() == rock_name.lower():
            matches.append((em_lbl, cn, ci))
            break

print(f'\n{len(matches)} matched pairs out of {len(em_labels)} emcal spectra:\n')
for em_lbl, cn, _ in matches:
    print(f'  {em_lbl!r:40s}  ↔  {cn!r}')

if not matches:
    raise SystemExit('No matches found — check label formats.')

# ---------------------------------------------------------------------------
# Interactive plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
fig.subplots_adjust(top=0.87, bottom=0.12, left=0.09, right=0.97)

idx = [0]   # mutable so the key handler can modify it


def draw(i: int) -> None:
    em_lbl, cn, ci = matches[i]
    ax.cla()

    ax.plot(em_xaxis, em_data[em_lbl], color='C0', lw=1.4,
            label=f'Measured  ({em_lbl})')
    ax.plot(cse_xaxis, cse_data[ci], color='C1', lw=1.2, ls='--',
            label=f'CSE library  ({cn})')

    ax.set_xlim(max(em_xaxis.max(), cse_xaxis.max()),
                min(em_xaxis.min(), cse_xaxis.min()))   # wavenumber: high→low
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Emissivity')
    ax.legend(fontsize=9)
    ax.set_title(f'{cn}  [{i + 1} / {len(matches)}]', fontsize=11)

    # secondary wavelength axis (only when xlim is valid)
    for sax in [c for c in ax.get_children()
                if hasattr(c, '_axis_below') and c is not ax]:
        try:
            sax.remove()
        except Exception:
            pass
    lo, hi = ax.get_xlim()
    if lo > 0 and hi > 0:
        secax = ax.secondary_xaxis('top', functions=(lambda x: 1e4/x, lambda x: 1e4/x))
        secax.xaxis.set_ticks([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25])
        secax.set_xlabel('Wavelength (µm)')

    fig.canvas.draw_idle()


def on_key(event) -> None:
    if event.key == 'right':
        idx[0] = (idx[0] + 1) % len(matches)
        draw(idx[0])
    elif event.key == 'left':
        idx[0] = (idx[0] - 1) % len(matches)
        draw(idx[0])
    elif event.key in ('q', 'escape'):
        plt.close(fig)


# ---------------------------------------------------------------------------
# Save all plots
# ---------------------------------------------------------------------------

OUT_DIR = ROOT / 'debug' / 'wards_comparison'
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'\nSaving {len(matches)} plots to {OUT_DIR} ...')

for i, (em_lbl, cn, ci) in enumerate(matches):
    draw(i)
    fig.canvas.draw()
    safe_name = cn.replace('/', '-').replace(' ', '_')
    out_path = OUT_DIR / f'{i+1:02d}_{safe_name}.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'  {out_path.name}')

print('Done.')

# ---------------------------------------------------------------------------
# Interactive navigation
# ---------------------------------------------------------------------------

fig.canvas.mpl_connect('key_press_event', on_key)
draw(0)
print('\nUse  ←/→  to navigate,  q  to quit.')
plt.show()
