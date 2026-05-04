#!/usr/bin/env python3
"""
Test horizontal merge of MIR and FIR emcal outputs.

Runs emcal on matched MIR and FIR datasets, merges them with the horizontal
path of merge(), then plots:
  - Top panel  : one selected sample before the merge (MIR solid, FIR dashed,
                 same colour) to visualise the spectral join in the overlap zone.
  - Bottom panel : all merged emissivity spectra, separated by PLOT_OFFSET for
                   legibility.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from speclab import emcal
from speclab.functions import merge
from speclab.plot import _PLOT_COLORS, _add_top_axis

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_DIR       = '~/Desktop/SpeclabDemo/2025-12-15'
MIR_DIR        = DATA_DIR + '/MIR'
FIR_DIR        = DATA_DIR + '/FIR'

METHOD         = 'nem'          # emissivity retrieval method
PREVIEW_SAMPLE = 'quartz'       # sample shown in the top panel
PLOT_OFFSET    = 0.3            # vertical offset between spectra in bottom panel

MIR_WN_RANGE   = (400.0, 1800.0)
FIR_WN_RANGE   = (100.0, 550.0)   # NEM optimisation window within FIR data range

# ---------------------------------------------------------------------------
# Step 1 — emcal
# ---------------------------------------------------------------------------
print("Running MIR emcal …")
out_mir = emcal(
    fdir=MIR_DIR,
    lab='nau',
    method=METHOD,
    wn_range=MIR_WN_RANGE,
    save=False,
    plot=False,
)

print("Running FIR emcal …")
out_fir = emcal(
    fdir=FIR_DIR,
    lab='nau',
    method=METHOD,
    wn_range=FIR_WN_RANGE,
    fir=True,
    save=False,
    plot=False,
)

# ---------------------------------------------------------------------------
# Step 2 — horizontal merge
# ---------------------------------------------------------------------------
print("Merging MIR + FIR …")
out_merged = merge(out_mir, out_fir, how='horizontal', align_overlap=True)

wn_merged = np.asarray(out_merged['xaxis'])
labels     = sorted(out_merged['emiss'].keys())
print(f"Merged xaxis: {wn_merged.min():.1f} – {wn_merged.max():.1f} cm⁻¹, "
      f"{len(wn_merged)} points")
print(f"Samples: {labels}")

# ---------------------------------------------------------------------------
# Step 3 — plot
# ---------------------------------------------------------------------------
wn_mir = np.asarray(out_mir['xaxis'])
wn_fir = np.asarray(out_fir['xaxis'])
overlap_lo = max(wn_mir.min(), wn_fir.min())
overlap_hi = min(wn_mir.max(), wn_fir.max())

fig = plt.figure(figsize=(12, 10))
gs  = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1, 3])

# ── Top panel : preview sample before merge ──────────────────────────────────
ax_top = fig.add_subplot(gs[0])

preview_color = _PLOT_COLORS[0]
if PREVIEW_SAMPLE in out_mir['emiss'] and PREVIEW_SAMPLE in out_fir['emiss']:
    ax_top.plot(wn_mir, out_mir['emiss'][PREVIEW_SAMPLE],
                color=preview_color, lw=1.2, ls='-',
                label=f'{PREVIEW_SAMPLE} — MIR')
    ax_top.plot(wn_fir, out_fir['emiss'][PREVIEW_SAMPLE],
                color=preview_color, lw=1.2, ls='--',
                label=f'{PREVIEW_SAMPLE} — FIR')
    ax_top.axvspan(overlap_lo, overlap_hi, color='gray', alpha=0.12,
                   label=f'Overlap ({overlap_lo:.0f}–{overlap_hi:.0f} cm⁻¹)')
else:
    ax_top.text(0.5, 0.5, f'Sample "{PREVIEW_SAMPLE}" not found in both outputs',
                ha='center', va='center', transform=ax_top.transAxes)

ax_top.set_xlim(max(wn_mir.max(), wn_fir.max()) + 50,
                min(wn_mir.min(), wn_fir.min()) - 50)
ax_top.set_ylabel('Emissivity')
ax_top.set_title(f'{PREVIEW_SAMPLE} — before merge (MIR solid, FIR dashed)')
ax_top.legend(fontsize=8)
_add_top_axis(ax_top)

# ── Bottom panel : all merged spectra, offset for visibility ─────────────────
ax_bot = fig.add_subplot(gs[1])

ytick_positions = []
ytick_labels    = []

for i, label in enumerate(labels):
    offset = i * PLOT_OFFSET
    ax_bot.plot(wn_merged, np.asarray(out_merged['emiss'][label]) + offset,
                color=_PLOT_COLORS[i % len(_PLOT_COLORS)], lw=1.2)
    ytick_positions.append(offset + 1.0)
    ytick_labels.append(label)

ax_bot.set_xlim(wn_merged.max() + 50, wn_merged.min() - 50)
ax_bot.set_xlabel('Wavenumber (cm⁻¹)')
ax_bot.set_ylabel('Emissivity (offset)')
ax_bot.set_title('Merged emissivity (MIR + FIR)')
ax_bot.set_yticks(ytick_positions)
ax_bot.set_yticklabels(ytick_labels, fontsize=9)
ax_bot.axvline(overlap_lo, color='gray', lw=0.8, ls=':', zorder=0)
ax_bot.axvline(overlap_hi, color='gray', lw=0.8, ls=':', zorder=0)
_add_top_axis(ax_bot)

fig.suptitle('MIR + FIR horizontal merge — emcal test', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('/tmp/test_merge.png', dpi=150)
print("Figure saved to /tmp/test_merge.png")
plt.show()