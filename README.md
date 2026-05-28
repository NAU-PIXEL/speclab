# speclab

Python spectroscopy processing library for thermal infrared emissivity calibration,
FTIR transmission/reflectance calibration, and VSWIR reflectance analysis.
Ports and extends algorithms from the DaVinci spectroscopy environment.

## Features

**Emission calibration**
- `emcal` — full lab emission calibration pipeline (blackbody IRF, NEM / hullfit / alpha / MMD emissivity retrieval, optional noise-free IRF smoothing, water-vapour correction)
- `dehyd` — water-vapour correction applied during or after `emcal`

**Emissivity retrieval methods**
- Normalized Emissivity Method (`nem`) — iterative BT peak anchoring
- Convex-hull Planck mixture (`hullfit` / `hullfit_linear`) — strict upper-bound enforcement
- Alpha Residuals (`alpha`) — mean-BT reference with max-emissivity rescaling
- Maximum–Minimum Difference (`mmd`) — simple contrast-based baseline

**FTIR transmission / reflectance calibration**
- `tracal` — transmission calibration from an AutomateFTIR measurement folder; pairs each sample with its closest-in-time background and blank
- `refcal` — reflectance calibration; same pipeline as `tracal`, reflectance semantics

**VSWIR reflectance analysis**
- `remove_continuum` — convex-hull continuum removal
- `band_parameters` — per-feature band depth, FWHM, area, asymmetry
- `smooth_spectrum` — Savitzky-Golay, boxcar, Gaussian smoothing
- `detect_bands` — automatic absorption feature detection with preset matching

**Spectral analysis**
- `sma` — spectral mixture analysis (NNLS, grouped endmembers, slope endmember, cumulative overlay plots)
- `merge` — combine emcal outputs vertically (same range, new samples) or horizontally (same samples, extended spectral range with DC-offset alignment)
- `match` — merge two individual spectra covering overlapping but different spectral ranges

**Data I/O**
- `readDVhdf` / `saveDVhdf` — DaVinci-format HDF5 read/write
- `dv_to_album` — normalise any DV HDF5 layout to the per-spectrum album dict
- `loadReflectanceCSV` / `saveReflectanceCSV` — wide-format VSWIR reflectance CSV
- `loadASD` — ASD ViewSpecPro tab-separated text export
- `convert_speclib` — command-line tool to convert spectral libraries: DV HDF5 → SpectralViewer format, USGS splib07 → speclab HDF5, CRISM spectral library → speclab HDF5

**Instrument presets** for NAU, ASU, SwRI lab spectrometers and TES / mTES satellite instruments

## GUI tools

| Command / Script        | Module                | Platform     | Purpose |
|-------------------------|-----------------------|--------------|---------|
| `speclib-viewer`        | `SpeclibViewer`       | all          | Browse and build spectral libraries (LWIR / VSWIR) |
| `emission-processor`    | `EmissionLWIR`        | all          | Interactive emcal / SMA results viewer |
| `reflectance-vswir`     | `ReflectanceVSWIR`    | all          | VSWIR reflectance viewer and band analysis |
| `AutomateFTIR.pyw`      | —                     | Windows only | Automated FTIR data collection (OMNIC DDE + Keithley 2700) |

## Installation

### Using conda / mamba (recommended)

```bash
conda create -n speclab python=3.11
conda activate speclab
pip install -e .
```

### Using venv

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Both approaches install all core dependencies listed in `pyproject.toml` and
make the `speclib-viewer`, `emission-processor`, and `reflectance-vswir`
entry-point commands available inside the environment.

For the `AutomateFTIR` GUI on Windows, install additional hardware drivers
after activating the environment:

```bash
pip install pyvisa pywin32
```

## Quick start

```python
import speclab

# Full emission calibration (reads BB temps from measurement-info.csv)
out = speclab.emcal('/path/to/data/', lab='nau', method='nem')

# Access results
wn     = out['xaxis']          # wavenumber axis (cm⁻¹)
emiss  = out['emiss']          # dict: sample label → emissivity array
labels = out['label']          # ordered list of sample labels

# Spectral mixture analysis
sma_out = speclab.sma(out, endlib='/path/to/library.hdf')

# Merge MIR and FIR emcal runs for the same samples
broadband = speclab.merge(out_mir, out_fir, how='horizontal')

# Transmission calibration from an AutomateFTIR folder
tra_out = speclab.tracal('/path/to/tracal_data/')

# Reflectance calibration
ref_out = speclab.refcal('/path/to/refcal_data/')

# Load and normalise a DV-format spectral library
raw   = speclab.readDVhdf('/path/to/library.hdf')
album = speclab.dv_to_album(raw)
```

### VSWIR reflectance

```python
from speclab import remove_continuum, band_parameters, smooth_spectrum

# wl: wavelength axis in nm, refl: reflectance array (0–1)
refl_sm = smooth_spectrum(wl, refl, method='savgol', window_nm=20, polyorder=3)
cr      = remove_continuum(wl, refl_sm, wl_range=(400, 2500))

# Band depth, centre, FWHM for the 2200 nm Al-OH feature
bp = band_parameters(wl, cr, wl_range=(2150, 2250))
print(bp['depth'], bp['center'], bp['fwhm'])
```

## Instrument presets

```python
list(speclab.INSTRUMENT_PRESETS.keys())
# ['spectrometer', 'nau', 'asu', 'swri', 'tes', 'tes5', 'mtes']
```

## Running the demo

`demo.py` walks through the full pipeline — emcal → SMA → plots — using the
bundled `example_data/` and `spectral_libraries/` folders.

```bash
# Install the package first (only needed once)
pip install -e .

# Run with default settings (single-folder, NEM, rock-forming mineral library)
python demo.py
```

Key settings at the top of `demo.py`:

| Variable | Default | Description |
|---|---|---|
| `USE_MULTI_FOLDER` | `False` | `True` runs emcal on all four `example_data/` subfolders and merges |
| `METHOD` | `'nem'` | Emissivity retrieval method: `'nem'`, `'hullfit'`, or `'mmd'` |
| `ENDLIB_PATH` | `spectral_libraries/speclib_JFS_rock_forming_minerals.hdf` | Endmember library for SMA |
| `USE_SPECLIBVIEWER` | `False` | `True` opens `SpeclibViewer` to build a custom library interactively |
| `SHOW_PLOTS` | `True` | Display emcal and SMA result plots |
| `SAVE_RESULTS` | `False` | Write HDF5 + CSV outputs alongside the data |

### Example data

`example_data/` contains:

**LWIR emission (Ward's rock standards, NAU)**

| Subfolder | Rock types |
|---|---|
| `WardRocks_igneous1` | Granite, granodiorite, diorite, gabbro, andesite |
| `WardRocks_igneous2` | Additional igneous standards |
| `WardRocks_metamorphic` | Metamorphic standards |
| `WardRocks_sedimentary` | Sedimentary standards |

Each subfolder contains sample CSVs, `bbhot.CSV`, `bbwarm.CSV`, and a
`measurement-info.csv` notes file with timestamps and instrument channel data.

**VSWIR reflectance**

| Path | Contents |
|---|---|
| `ASD_data/Orochi_analyses_2.csv` | ASD field reflectance spectra (Orochi site) |
| `dummy_vswir/` | Synthetic VSWIR dataset for testing |

## Bundled spectral libraries

| File | Spectral range | Contents |
|---|---|---|
| `speclib_JFS_rock_forming_minerals.hdf` | LWIR | NAU rock-forming mineral endmember library |
| `ASU_speclib_full_SV.hdf` | LWIR | ASU thermal emission spectral library |
| `CSE_wards_rock_library_SV.hdf` | LWIR | CSE Ward's rock standards |
| `usgs_splib07_cvASD.hdf` | VSWIR | USGS Spectral Library 7 — ASD-convolved variant |
| `crism_speclib.hdf` | VSWIR | CRISM spectral library |

## Spectral library conversion

### DV HDF5 → SpectralViewer format
```bash
python -m speclab.convert_speclib convert input.hdf output.hdf
python -m speclab.convert_speclib test    input.hdf
```

### USGS splib07 → speclab HDF5
```bash
python -m speclab.convert_speclib usgs splib07_dir/ output.hdf
# Restrict to specific chapters and library variant:
python -m speclab.convert_speclib usgs splib07_dir/ output.hdf --source cvASD --chapters Minerals Vegetation
```

### CRISM spectral library → speclab HDF5
```bash
python -m speclab.convert_speclib crism input.hdf output.hdf
```
