# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Auditory neural recording analysis pipeline (Feldheim Lab, UC Santa Cruz). Processes electrophysiological spike-sorted data to identify auditory-responsive neurons and fit spatial receptive fields to statistical distributions. Originally ported from MATLAB (see `pipeline_organization.md` for the original structure).

## Running the Pipeline

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

Key arguments:
- `-i`: Path to spike-sorted output directory (kilosort4 or VISION)
- `-p`: Probe type (`npxl`, `A`, or `AN`)
- `-ss`: Spike sorter used (`kilosort4` or `vision`)
- `-f`: Sampling rate in Hz (default 30000)
- `-c`: Column name in cluster TSV for neuron classification
- `-class`: Which classification to include (`good`, `mua`, `all`)

**Before running**: Edit `config.py` to set `stimdir` (path to stimulus files) and `stim_dict` (mapping of data segments to stimulus files, distribution types, and multipliers).

## Architecture

### Pipeline Flow

```
Spike-sorted data (kilosort4 .npy/.tsv or VISION .mat)
  → data_load() class [auditory_analysis_pipeline.py]
    → Load spikes, TTLs, channel maps, convert to ASDF format

For each stimulus segment defined in config.stim_dict:
  → getTTLseg() → readStimFile() → patternGen()  [preprocessing.py]
    → Generate spike rasters aligned to stimulus onsets (5D array)
  → sigAudFRCompareSpont()  [preprocessing.py]
    → Statistical test: stimulus-evoked vs spontaneous firing
    → Supports poisson, quasipoisson, negative binomial
  → PatternToCount()  [preprocessing.py]
    → Convert to firing rate histograms
  → kent_fit() + uniform_fit()  [distributions_fit.py]
    → Fit Kent distribution on sphere for spatial receptive fields
    → Compare with uniform model via AIC/BIC
  → Save results as .npy arrays and .csv
```

### Module Responsibilities

- **`auditory_analysis_pipeline.py`** - Main entry point. Argument parsing, orchestrates the full pipeline, contains `data_load()` class for loading spike-sorted data from kilosort4 or VISION formats.
- **`config.py`** - `configs()` class centralizing all tunable parameters: stimulus mappings, analysis windows, statistical test settings, significance thresholds.
- **`python/preprocessing.py`** - Data processing: probe maps (`probeMap`), TTL extraction (`ttl_rise`), stimulus file parsing (`readStimFile`), spike pattern generation (`patternGen`), firing rate computation (`PatternToCount`), statistical significance tests (`sigAudFRCompareSpont`, `sigAudFR_zeta_pvalue`).
- **`python/distributions_fit.py`** - Distribution fitting classes: `kent_fit` (primary, fits Kent distribution on sphere with 50 random starts), `gaussian_fit`, `vonMises_fit`, `uniform_fit`. Includes AIC/BIC model comparison functions.
- **`python/visualizations.py`** - Plotting: receptive field data loading (`load_RF_data`), probe layout plots, raster plots (`PatternRaster3d`), PSTHs, model performance comparison, cluster info displays.
- **`python/hdf5manager.py`** - HDF5 file I/O utility class. Also has a CLI for merging/copying/extracting HDF5 files.

### Import Pattern

Modules in `python/` are imported via `sys.path.append('./python')` with wildcard imports (`from preprocessing import *`). Run the pipeline from the repository root.

## Key Data Structures

- **ASDF**: Array of Spike Data Format - `np.array` of dtype `object` where each element is a list/array of spike times (in ms) for one neuron.
- **Pattern**: 5D numpy array `[neurons, elevation, azimuth, repetitions, time_bins]` - spike rasters aligned to stimulus onsets.
- **`datasep`**: Array of timestamps marking boundaries between data segments (used with `seg` index to extract segment-specific TTLs).

## Output Structure

Results save to `{parent_of_input}/results/`:
```
results/
├── data_processing_log.txt
├── images/                          (if --plot)
└── {segment} mult {multiplier}/
    ├── pattern.npy, good_neurons.npy, activity-df.csv
    ├── aud_neurons_qpoisson.npy, windows.npy
    └── {distribution} fit win {window}/
        ├── parameters.npy, variances.npy, neuron_assess.npy
        ├── aic_bic.npy, datafits.npy, modelfits.npy, sumresid.npy
```

## Dependencies

numpy, scipy, pandas, matplotlib, h5py, zetapy. No requirements.txt exists; install manually.

## Incomplete Features

- `python/random_chord_analysis.py` is empty. RandomChord analysis is stubbed out in the main pipeline (lines ~487-526) with commented-out code. Reference MATLAB implementation exists at `python/DoRandomChordAnalysis.m`.

## Development Notes

- No automated tests. Development notebooks in `scratch/` (numbered sequentially) are used for prototyping.
- Tabs are used for indentation throughout the codebase.
- The project uses bare `except` clauses in several places for error handling.
