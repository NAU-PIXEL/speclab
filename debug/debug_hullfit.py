#!/usr/bin/env python3
"""
Side-by-side comparison of NEM, emissivity_hullfit (old), emissivity_hullfit (new),
hullfit_linear (closed-form n_bb=2), and emissivity_alpha.

Runs full emcal processing (radiance calibration + downwelling correction from
notes) via emcal(method='nem'), then calls all five emissivity functions on
each sample using the same calibrated radiance and downwelling temperature.

Two-panel figure per sample:
  Panel 1 — calibrated radiance + BB models for all methods
  Panel 2 — emissivity comparison
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from speclab import emcal, utils
from speclab.functions import (
    emissivity_nem,
    emissivity_alpha,
    emissivity_hullfit,
    emissivity_hullfit_linear,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

# FDIR          = os.path.expanduser('~') + '/Nextcloud/107 Storage/FTIR Data/jfsmekens/FIR_project/2025-12-15/MIR'
FDIR          = os.path.expanduser('~') + '/Desktop/SpeclabDemo/data01'
WN_RANGE_NEM  = (400.0, 1600.0)
WN_RANGE_HULL = (250.0, 2000.0)
WN_RANGE_ALPHA = (500.0, 1700.0)
MAX_EMISS     = 1.0
N_BB          = 3

VIOLATION_WEIGHT  = 5.0
VIOLATION_TOL     = 0.0
ESCALATION_FACTOR = 4.0
MAX_ESCALATIONS   = 4
TEMP_HALFWIDTH    = 100.0

# ---------------------------------------------------------------------------
# Step 1 — full emcal (NEM) to get calibrated radiance + downwelling temps
# ---------------------------------------------------------------------------
print("Running emcal (NEM) for radiance calibration …")
em_out = emcal(
    fdir=FDIR,
    lab='nau',
    method='nem',
    max_emiss=MAX_EMISS,
    save=False,
    plot=False,
)

wn     = np.asarray(em_out['xaxis'])
labels = em_out['label']
print(f"wn : {wn[0]:.1f} – {wn[-1]:.1f} cm⁻¹  ({len(wn)} channels)")
print(f"{len(labels)} sample(s) found — processing first: {labels[0]}")

for i_l in range(len(labels)):
    label = labels[i_l]
    data  = em_out['rad'][label].copy()

    nem_pre    = em_out['emiss_full'][label]
    dw_t       = float(nem_pre['downwelling_t'])
    dw_e       = float(nem_pre['downwelling_e'])
    dw_rad_arr = nem_pre['downwelling_rad']
    if dw_rad_arr is None:
        dw_rad_arr = np.zeros(len(wn))

    print(f"\n{'='*60}")
    print(f"Sample: {label}   dw_t = {dw_t:.1f} K  (e = {dw_e:.3f})")
    print(f"{'='*60}")

    # ---------------------------------------------------------------------------
    # Step 2 — run all five emissivity methods on the same data
    # ---------------------------------------------------------------------------
    print("\nRunning emissivity_nem …")
    r_nem = emissivity_nem(
        wn, data,
        inst='nau',
        max_emiss=MAX_EMISS,
        wn_range=WN_RANGE_NEM,
        downwelling_t=dw_t,
        downwelling_e=dw_e,
    )
    print(f"  temp     : {r_nem['temp']:.1f} K")

    print("\nRunning emissivity_alpha …")
    r_alpha = emissivity_alpha(
        wn, data,
        wn_range=WN_RANGE_ALPHA,
        max_emiss=MAX_EMISS,
        downwelling_t=dw_t,
        downwelling_e=dw_e,
    )
    print(f"  t_ref    : {r_alpha['t_ref']:.1f} K")
    print(f"  temp     : {r_alpha['temp']:.1f} K")
    print(f"  elapsed  : {r_alpha['elapsed']:.4f} s")

    print("\nRunning emissivity_hullfit …")
    r2 = emissivity_hullfit(
        wn, data,
        n_bb=N_BB,
        wn_range=WN_RANGE_HULL,
        max_emiss=MAX_EMISS,
        downwelling_t=dw_t,
        downwelling_e=dw_e,
        violation_weight=VIOLATION_WEIGHT,
        violation_tol=VIOLATION_TOL,
        escalation_factor=ESCALATION_FACTOR,
        max_escalations=MAX_ESCALATIONS,
        temp_halfwidth=TEMP_HALFWIDTH,
    )
    print(f"  temp_range         : {r2['temp_range']}")
    print(f"  bb_temps           : {[f'{t:.1f}' for t in r2['bb_temps']]} K")
    print(f"  bb_fracs           : {[f'{f:.3f}' for f in r2['bb_fracs']]}")
    print(f"  n_violations_final : {r2['n_violations_final']}")
    print(f"  elapsed            : {r2['elapsed']:.3f} s")

    print("\nRunning emissivity_hullfit_linear (closed-form, n_bb=2) …")
    r3 = emissivity_hullfit_linear(
        wn, data,
        wn_range=WN_RANGE_HULL,
        max_emiss=MAX_EMISS,
        downwelling_t=dw_t,
        downwelling_e=dw_e,
        temp_halfwidth=TEMP_HALFWIDTH,
    )
    print(f"  temp_range : {r3['temp_range']}")
    print(f"  bb_temps   : {[f'{t:.1f}' for t in r3['bb_temps']]} K")
    print(f"  bb_fracs   : {[f'{f:.3f}' for f in r3['bb_fracs']]}")
    fit_mask3   = (r3['wn0'] >= WN_RANGE_HULL[0]) & (r3['wn0'] <= WN_RANGE_HULL[1])
    n_viol_hf2  = int((r3['model0'][fit_mask3] < r3['data0_eff'][fit_mask3]).sum())
    print(f"  n_violations (post-solve) : {n_viol_hf2}  (expected 0)")
    print(f"  elapsed    : {r3['elapsed']:.4f} s  (vs hullfit new: {r2['elapsed']:.3f} s,  "
          f"speedup ≈ {r2['elapsed'] / r3['elapsed']:.0f}×)")

    # ---------------------------------------------------------------------------
    # Step 3 — figure
    # Radiance panel uses data0_eff space: (data - dw_rad) / max_emiss
    # alpha rad_bb is B(T_ref) — not a bounding model, just the reference BB
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=[17, 6])
    fig.suptitle(f"Emissivity method comparison — {label}  |  dw_t = {dw_t:.1f} K", fontsize=11)

    def fw(x):
        return 10000 / x

    wn0         = r2['wn0']
    data0       = r2['data0_eff']
    model_nem   = (r_nem['rad_bb']   - dw_rad_arr) / MAX_EMISS
    model_alpha = (r_alpha['rad_bb'] - dw_rad_arr) / MAX_EMISS
    model2      = r2['model0']
    model3      = r3['model0']
    fit_mask    = (wn0 >= WN_RANGE_HULL[0]) & (wn0 <= WN_RANGE_HULL[1])

    n_viol_nem = int((model_nem[fit_mask] < data0[fit_mask]).sum())

    # Panel 1 — radiance + BB models
    ax = axes[0]
    ax.plot(wn0, data0,       c='k',             lw=1.2, label='Data (dw-corr / max_emiss)', zorder=3)
    ax.plot(wn0, model_nem,   c='forestgreen',  lw=1.4,
            label=f'NEM         T={r_nem["temp"]:.1f} K  ({n_viol_nem} viol.)', zorder=4)
    ax.plot(wn0, model_alpha, c='mediumorchid', lw=1.4, ls=':',
            label=f'alpha       T_ref={r_alpha["t_ref"]:.1f} K  T={r_alpha["temp"]:.1f} K', zorder=4)
    ax.plot(wn0, model2,      c='tomato',       lw=1.4,
            label=f'hullfit     T={r2["temp"]:.1f} K  ({r2["n_violations_final"]} viol.)', zorder=4)
    ax.plot(wn0, model3,      c='darkorange',   lw=1.4, ls='--',
            label=f'hullfit_lin T={r3["temp"]:.1f} K  ({n_viol_hf2} viol.)', zorder=4)

    for model, color in [(model_nem, 'forestgreen'), (model2, 'tomato'), (model3, 'darkorange')]:
        ax.fill_between(wn0, model, data0, where=(model < data0),
                        color=color, alpha=0.15, zorder=2)

    ax.scatter(r2['wn'], r2['data'], c='tomato', s=4, alpha=0.5, zorder=5,
               label=f'HF fit set ({len(r2["wn"])} pts)')
    ax.set(xlabel='Wavenumber [cm⁻¹]',
           ylabel=r'Spectral Radiance [$W / (m^2 \cdot sr \cdot cm^{-1})$]')
    ax.invert_xaxis()
    ax.legend(fontsize=7)
    secax = ax.secondary_xaxis('top', functions=(fw, fw))
    secax.xaxis.set_ticks([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    secax.set_xlabel(r'Wavelength [$\mu$m]')

    # Panel 2 — emissivity
    ax = axes[1]
    ax.plot(wn0, r_nem['emiss'],   c='forestgreen',  lw=1.4,
            label=f'NEM         T={r_nem["temp"]:.1f} K')
    ax.plot(wn0, r_alpha['emiss'], c='mediumorchid', lw=1.4, ls=':',
            label=f'alpha       T={r_alpha["temp"]:.1f} K  (T_ref={r_alpha["t_ref"]:.1f} K)')
    ax.plot(wn0, r2['emiss'],      c='tomato',       lw=1.4,
            label=f'hullfit     T={r2["temp"]:.1f} K')
    ax.plot(wn0, r3['emiss'],      c='darkorange',   lw=1.4, ls='--',
            label=f'hullfit_lin T={r3["temp"]:.1f} K')
    ax.axhline(1.0, c='k', ls='-', lw=0.8)
    ax.set(xlabel='Wavenumber [cm⁻¹]', ylabel='Emissivity')
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    secax = axes[1].secondary_xaxis('top', functions=(fw, fw))
    secax.xaxis.set_ticks([2, 3, 4, 5, 6, 8, 10, 12, 15, 20])
    secax.set_xlabel(r'Wavelength [$\mu$m]')

    fig.tight_layout()
    plt.show()
