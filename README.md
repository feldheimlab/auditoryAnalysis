# auditoryAnalysis

Auditory neural recording analysis pipeline (Feldheim Lab, UC Santa Cruz). Processes electrophysiological spike-sorted data to identify auditory-responsive neurons and fit spatial receptive fields to statistical distributions.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. Dependencies: numpy, scipy, pandas, matplotlib, h5py, zetapy.

## Usage

1. Edit `config.py` to set `stimdir` (path to stimulus files) and `stim_dict` (mapping of data segments to stimulus files, distribution types, and multipliers).

2. Run the pipeline from the repository root:

```bash
python auditory_analysis_pipeline.py \
  -i /path/to/spike-sorted-output \
  -p npxl \
  -ss kilosort4 \
  -f 30000 \
  --plot \
  -c group \
  -class good
```

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-i` | Path to spike-sorted output directory | *(required)* |
| `-p` | Probe type: `npxl`, `A`, or `AN` | `npxl` |
| `-ss` | Spike sorter: `kilosort4` or `vision` | `kilosort4` |
| `-f` | Sampling rate in Hz | `30000` |
| `-c` | Column name in cluster TSV for neuron classification | `group` |
| `-class` | Which classification to include: `good`, `mua`, `all` | `good` |
| `--plot` | Save output plots | off |

## Pipeline Overview

```
Spike-sorted data (kilosort4 .npy/.tsv or VISION .mat)
  → Load spikes, TTLs, channel maps, convert to ASDF format

For each stimulus segment in config.stim_dict:
  → Extract segment TTLs → parse stimulus file → generate spike rasters
  → Statistical test: evoked vs spontaneous firing (poisson/quasipoisson/nbinom)
  → Convert to firing rate histograms
  → Fit Kent distribution on sphere for spatial receptive fields
  → Compare with uniform model via AIC/BIC
  → Save results as .npy arrays and .csv
```

For random chord stimuli, the pipeline computes spike-triggered averages (STAs) to characterize spectrotemporal receptive fields. Enable by setting the fit type to `'RandomChord'` in `config.stim_dict`.

## Project Structure

```
auditory_analysis_pipeline.py   Main entry point
config.py                       Tunable parameters (stimulus mappings, windows, thresholds)
python/
├── __init__.py
├── preprocessing.py            Probe maps, TTL extraction, spike patterns, significance tests
├── distributions_fit.py        Kent, Gaussian, von Mises, uniform distribution fitting
├── visualizations.py           Receptive field plots, rasters, PSTHs, model comparison
├── random_chord_analysis.py    STA analysis for random chord stimuli
└── hdf5manager.py              HDF5 file I/O utility
scratch/                        Development notebooks
```

## Output

Results save to `{parent_of_input}/results/`:

```
results/
├── data_processing_log.txt
├── images/                              (if --plot)
└── {segment} mult {multiplier}/
    ├── pattern.npy, good_neurons.npy, activity-df.csv
    ├── aud_neurons_qpoisson.npy, windows.npy
    └── {distribution} fit win {window}/
        ├── parameters.npy, variances.npy, neuron_assess.npy
        ├── aic_bic.npy, datafits.npy, modelfits.npy, sumresid.npy
```