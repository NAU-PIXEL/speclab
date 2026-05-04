# speclab

Python spectroscopy processing library for thermal infrared emissivity calibration
and analysis. Ports and extends algorithms from the DaVinci spectroscopy environment.

## Features

**Calibration**
- `emcal` — full lab emission calibration pipeline (blackbody IRF, NEM / hullfit / alpha / MMD emissivity retrieval, optional noise-free IRF smoothing)
- `tracal` — transmission calibration against a blank reference
- `dehyd` — water-vapour correction applied during or after `emcal`

**Emissivity retrieval methods**
- Normalized Emissivity Method (`nem`) — iterative BT peak anchoring
- Convex-hull Planck mixture (`hullfit` / `hullfit_linear`) — strict upper-bound enforcement
- Alpha Residuals (`alpha`) — mean-BT reference with max-emissivity rescaling
- Maximum–Minimum Difference (`mmd`) — simple contrast-based baseline

**Spectral analysis**
- `sma` — spectral mixture analysis (NNLS, grouped endmembers, slope endmember, cumulative overlay plots)
- `merge` — combine emcal outputs vertically (same range, new samples) or horizontally (same samples, extended spectral range with DC-offset alignment in the overlap)
- `match` — merge two individual spectra covering overlapping but different spectral ranges

**Data I/O**
- `readDVhdf` / `saveDVhdf` — DaVinci-format HDF5 read/write
- `dv_to_album` — normalise any DV HDF5 layout to the per-spectrum album dict expected by `sma` and SpectralViewer
- `convert_speclib` — command-line tool to convert a DV speclib HDF5 to SpectralViewer format

**Instrument presets** for NAU, ASU, SwRI lab spectrometers and TES / mTES satellite instruments

## GUI tools

| Command | Module | Purpose |
|---|---|---|
| `speclib-viewer-TIR` | `SpeclibViewerTIR` | Browse and compare spectral libraries |
| `emission-processor` | `EmissionProcessor` | Interactive emcal / SMA results viewer |
| *(planned)* | `EmissionAutomation` | Automated emission data collection GUI |

## Installation

```bash
pip install -e .
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

# Load and normalise a DV-format spectral library
raw   = speclab.readDVhdf('/path/to/library.hdf')
album = speclab.dv_to_album(raw)
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
| `USE_SPECLIBVIEWER` | `False` | `True` opens `SpeclibViewerTIR` to build a custom library interactively |
| `SHOW_PLOTS` | `True` | Display emcal and SMA result plots |
| `SAVE_RESULTS` | `False` | Write HDF5 + CSV outputs alongside the data |

### Example data

`example_data/` contains four Ward's rock standard sets measured at NAU:

| Subfolder | Rock types |
|---|---|
| `WardRocks_igneous1` | Granite, granodiorite, diorite, gabbro, andesite |
| `WardRocks_igneous2` | Additional igneous standards |
| `WardRocks_metamorphic` | Metamorphic standards |
| `WardRocks_sedimentary` | Sedimentary standards |

Each subfolder contains sample CSVs, `bbhot.CSV`, `bbwarm.CSV`, and a
`measurement-info.csv` notes file with timestamps and instrument channel data.

## Convert a DV spectral library to SpectralViewer format

```bash
python -m speclab.convert_speclib convert input.hdf output.hdf
python -m speclab.convert_speclib test    input.hdf
```
