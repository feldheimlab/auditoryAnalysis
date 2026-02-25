#!/usr/bin/env python3
"""Generate standard plots for RF or auditory-responsive neurons.

Produces topographic maps, cluster info/waveform panels, model performance
comparisons, and (optionally) raster plots for selected neurons.
"""

import argparse
import os
import sys

import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')

# Ensure the repo root is on the path so the `python` package resolves.
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from python.visualizations import (
    load_RF_data,
    cluster_info_waveform,
    model_performance,
    PatternRaster3d,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot standard figures for RF or auditory-responsive neurons.'
    )
    parser.add_argument(
        '-i', '--input', nargs='+', required=True,
        help='One or more data directories (segment results folders).',
    )
    parser.add_argument(
        '-s', '--selection', choices=['rf', 'auditory'], default='rf',
        help='Neuron selection mode (default: rf).',
    )
    parser.add_argument(
        '-w', '--window', type=int, default=0,
        help='Analysis window index (default: 0).',
    )
    parser.add_argument(
        '--min-fr', type=float, default=0,
        help='Minimum firing rate in Hz (default: 0, no filter).',
    )
    parser.add_argument(
        '--no-raster', action='store_true',
        help='Skip raster plots.',
    )
    parser.add_argument(
        '-ss', '--spikesorting', default='kilosort4',
        help='Spike sorter (default: kilosort4).',
    )
    parser.add_argument(
        '--anat-ref', type=float, default=0,
        help='Anatomical reference value (um) to shift probe span along x-axis (default: 0).',
    )
    return parser.parse_args()


def build_mask(data, selection, window):
    """Return a boolean mask over good_neurons for the requested selection."""
    w = window
    pflag = f'quasipoisson win {w} pos'
    nflag = f'quasipoisson win {w} neg'
    auditory = (data.activity_df[pflag] | data.activity_df[nflag]).values.astype(bool)

    if selection == 'rf':
        return data.is_rf[w] & auditory
    return auditory


def _get_probe_span(data, rotation_deg=15):
    """Return rotated physical span position (um) for each good neuron.

    The probe span (x-coordinate of channel_positions) is rotated about
    the top-left-most point of the probe by *rotation_deg* degrees so that
    the physical layout better reflects anatomical orientation.

    Parameters
    ----------
    data : load_RF_data
    rotation_deg : float
        Counter-clockwise rotation in degrees applied to probe coordinates.

    Returns
    -------
    span : np.ndarray
        Rotated span position for each neuron (length = len(good_neurons)).
    """
    cluster_dedup = data.cluster[~data.cluster.index.duplicated(keep='first')]
    channels = cluster_dedup.loc[data.good_neurons, 'most_active_channel'].values.astype(int)
    probe_x = data.channelposition[channels, 0]
    probe_y = data.channelposition[channels, 1]

    # Origin = top-left of probe (min x, max y)
    ox = data.channelposition[:, 0].min()
    oy = data.channelposition[:, 1].max()

    angle = np.radians(-rotation_deg)  # negative for counter-clockwise
    dx = probe_x - ox
    dy = probe_y - oy
    rotated_span = dx * np.cos(angle) - dy * np.sin(angle)
    return rotated_span


def weighted_linear_fit(x, y, y_err, n_starts=50):
    """Fit y = mx + b weighted by 1/y_err^2 with random restarts.

    Runs *n_starts* fits from random initial parameters and returns the
    result with the lowest weighted chi-squared.

    Parameters
    ----------
    x, y : array-like
        Data coordinates.
    y_err : array-like
        Standard errors on *y* used as weights (1/sigma^2).
    n_starts : int
        Number of random initial-guess attempts.

    Returns
    -------
    slope, intercept, slope_err, intercept_err : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.clip(np.asarray(y_err, dtype=float), 1e-6, None)

    def _line(xv, m, b):
        return m * xv + b

    def _chi2(popt):
        residuals = (y - _line(x, *popt)) / sigma
        return np.sum(residuals ** 2)

    # Estimate reasonable parameter ranges for random starts
    y_range = y.max() - y.min() if y.max() != y.min() else 1.0
    x_range = x.max() - x.min() if x.max() != x.min() else 1.0

    best_chi2 = np.inf
    best_popt = None
    best_pcov = None

    for i in range(n_starts):
        if i == 0:
            p0 = None  # let curve_fit use default (1, 1)
        else:
            m0 = np.random.uniform(-y_range / x_range * 2, y_range / x_range * 2)
            b0 = np.random.uniform(y.min() - y_range, y.max() + y_range)
            p0 = [m0, b0]
        try:
            popt, pcov = curve_fit(_line, x, y, p0=p0, sigma=sigma,
                                   absolute_sigma=True)
            chi2 = _chi2(popt)
            if chi2 < best_chi2:
                best_chi2 = chi2
                best_popt = popt
                best_pcov = pcov
        except RuntimeError:
            continue

    if best_popt is None:
        raise RuntimeError('All fit attempts failed')

    perr = np.sqrt(np.diag(best_pcov))
    return best_popt[0], best_popt[1], perr[0], perr[1]


def plot_topographic_map(data, selected_mask, savepath, window, anat_ref=0):
    """Topographic map: azimuthal RF position vs rotated physical probe span."""
    params = data.parameters[window]
    sem_values = data.variances[window]

    azimuth = np.degrees(params[:, 3])           # phi → degrees
    azim_err = np.degrees(np.sqrt(np.abs(sem_values[:, 3])))
    span = (_get_probe_span(data) - anat_ref) / 1000  # um → mm

    fig, ax = plt.subplots(figsize=(8, 6))

    # Only plot selected (RF) neurons, coloured by source with azimuthal error bars
    cmap = plt.cm.tab10
    sources = data.source_indices[selected_mask]
    unique_sources = np.unique(sources)
    for src in unique_sources:
        src_sel = selected_mask & (data.source_indices == src)
        idxs = np.where(src_sel)[0]
        label = data.source_labels[src] if src < len(data.source_labels) else f'source {src}'
        ax.errorbar(
            span[idxs], azimuth[idxs],
            yerr=np.clip(azim_err[idxs], 0, 50),
            fmt='o', color=cmap(src), markersize=6,
            capsize=2, capthick=1, alpha=0.8,
            label=label,
        )

    # Weighted linear fit through selected neurons with span >= 0
    sel_span = span[selected_mask]
    sel_azim = azimuth[selected_mask]
    sel_err = azim_err[selected_mask]
    valid = np.isfinite(sel_err) & (sel_err > 0) & (sel_span >= 0) & (sel_azim >= -50)
    if valid.sum() >= 2:
        m, b, m_err, b_err = weighted_linear_fit(
            sel_span[valid], sel_azim[valid], sel_err[valid])
        x_fit = np.linspace(0, 1.5, 200)
        ax.plot(x_fit, m * x_fit + b, 'r-', linewidth=1.5,
                label=f'fit: {m:.1f}\u00b1{m_err:.1f} deg/mm')
        print(f'  Weighted linear fit: slope={m:.1f}\u00b1{m_err:.1f} deg/mm, '
              f'intercept={b:.1f}\u00b1{b_err:.1f} deg')

    ax.set_xlabel('A-P Position (mm)')
    ax.set_ylabel('Azimuthal RF Position (deg)')
    ax.set_xlim(0, 1.5)
    ax.set_ylim(-50, 150)
    datasets = ', '.join(data.source_labels)
    ax.set_title(f'Topographic Map — window {window}\n{datasets}')
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)


def plot_rf_comparison(dir_a, dir_b, window, spikesorting, min_fr, savepath):
    """Compare azimuthal RF positions of shared neurons across two segments.

    Loads each segment independently so that original cluster IDs are
    preserved, then matches neurons by cluster ID.
    """
    data_a = load_RF_data(dir_a, spikesorting=spikesorting)
    data_b = load_RF_data(dir_b, spikesorting=spikesorting)

    label_a = os.path.basename(os.path.normpath(dir_a))
    label_b = os.path.basename(os.path.normpath(dir_b))

    w = window
    pflag = f'quasipoisson win {w} pos'
    nflag = f'quasipoisson win {w} neg'

    # Build RF + auditory mask for each segment
    aud_a = (data_a.activity_df[pflag] | data_a.activity_df[nflag]).values.astype(bool)
    mask_a = data_a.is_rf[w] & aud_a
    aud_b = (data_b.activity_df[pflag] | data_b.activity_df[nflag]).values.astype(bool)
    mask_b = data_b.is_rf[w] & aud_b

    # Optional FR filter
    if min_fr > 0:
        cluster_a = data_a.cluster[~data_a.cluster.index.duplicated(keep='first')]
        fr_a = cluster_a.loc[data_a.good_neurons, 'fr'].values
        mask_a &= fr_a >= min_fr
        cluster_b = data_b.cluster[~data_b.cluster.index.duplicated(keep='first')]
        fr_b = cluster_b.loc[data_b.good_neurons, 'fr'].values
        mask_b &= fr_b >= min_fr

    # Map cluster ID → index for each segment
    id_to_idx_a = {int(cid): i for i, cid in enumerate(data_a.good_neurons)}
    id_to_idx_b = {int(cid): i for i, cid in enumerate(data_b.good_neurons)}

    rf_ids_a = set(int(c) for c in data_a.good_neurons[mask_a])
    rf_ids_b = set(int(c) for c in data_b.good_neurons[mask_b])

    both = sorted(rf_ids_a & rf_ids_b)
    only_a = rf_ids_a - rf_ids_b
    only_b = rf_ids_b - rf_ids_a

    print(f'\nRF comparison: {label_a} vs {label_b}')
    print(f'  RF in both segments: {len(both)}')
    print(f'  RF only in {label_a}: {len(only_a)}')
    print(f'  RF only in {label_b}: {len(only_b)}')

    if len(both) == 0:
        print('  No shared RF neurons to compare.')
        return

    params_a = data_a.parameters[w]
    params_b = data_b.parameters[w]
    var_a = data_a.variances[w]
    var_b = data_b.variances[w]

    azim_a = np.array([np.degrees(params_a[id_to_idx_a[c], 3]) for c in both])
    azim_b = np.array([np.degrees(params_b[id_to_idx_b[c], 3]) for c in both])
    err_a = np.array([np.degrees(np.sqrt(np.abs(var_a[id_to_idx_a[c], 3]))) for c in both])
    err_b = np.array([np.degrees(np.sqrt(np.abs(var_b[id_to_idx_b[c], 3]))) for c in both])

    fig, ax = plt.subplots(figsize=(7, 7))

    # Unity line
    lo = min(azim_a.min(), azim_b.min()) - 20
    hi = max(azim_a.max(), azim_b.max()) + 20
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='unity')

    ax.errorbar(
        azim_a, azim_b,
        xerr=np.clip(err_a, 0, 50),
        yerr=np.clip(err_b, 0, 50),
        fmt='o', markersize=6, capsize=2, capthick=1, alpha=0.8,
    )

    ax.set_xlabel(f'Azimuthal RF Position — {label_a} (deg)')
    ax.set_ylabel(f'Azimuthal RF Position — {label_b} (deg)')
    ax.set_title(
        f'RF Comparison — window {w}\n'
        f'shared: {len(both)}, only {label_a}: {len(only_a)}, only {label_b}: {len(only_b)}'
    )
    ax.legend(loc='best', fontsize='small')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)
    print(f'  Saved comparison → {savepath}')


def main():
    args = parse_args()

    # --- Load data ---
    input_dirs = args.input if len(args.input) > 1 else args.input[0]
    print(f'Loading data from {args.input} ...')
    data = load_RF_data(input_dirs, spikesorting=args.spikesorting)

    w = args.window
    mask = build_mask(data, args.selection, w)

    # Optional firing-rate filter
    if args.min_fr > 0:
        # cluster DF may have duplicate indices when multiple segments share
        # the same kilosort output; deduplicate before lookup.
        cluster_dedup = data.cluster[~data.cluster.index.duplicated(keep='first')]
        fr = cluster_dedup.loc[data.good_neurons, 'fr'].values
        mask &= fr >= args.min_fr

    selected_indices = np.where(mask)[0]
    selected_clusters = data.good_neurons[selected_indices]
    print(f'Selected {len(selected_indices)} neurons ({args.selection} mode, window {w}).')

    if len(selected_indices) == 0:
        print('No neurons matched the criteria — nothing to plot.')
        return

    # --- Output directory ---
    subdir = 'rf_neurons' if args.selection == 'rf' else 'auditory_neurons'
    outdir = os.path.join(data.image_save_loc, subdir)
    os.makedirs(outdir, exist_ok=True)
    print(f'Saving plots to {outdir}')

    # --- Topographic map ---
    topo_path = os.path.join(outdir, 'topographic_map.png')
    plot_topographic_map(data, mask, topo_path, w, anat_ref=args.anat_ref)
    print(f'  Saved topographic map → {topo_path}')

    # --- Per-neuron plots ---
    window_range = [int(data.windows[w][0]), int(data.windows[w][1])]
    n_total = len(selected_indices)

    for count, (idx, clust) in enumerate(zip(selected_indices, selected_clusters), 1):
        tag = f'[{count}/{n_total}] cluster {clust}'

        # cluster_info_waveform
        try:
            sp = os.path.join(outdir, f'cluster_info_cluster{clust}.png')
            cluster_info_waveform(data, cluster=int(clust), index=int(idx), savepath=sp)
            plt.close('all')
            print(f'  {tag}: cluster_info ✓')
        except Exception as e:
            print(f'  {tag}: cluster_info FAILED — {e}')

        # model_performance
        try:
            sp = os.path.join(outdir, f'model_performance_cluster{clust}.png')
            model_performance(data, index=int(idx), savepath=sp)
            plt.close('all')
            print(f'  {tag}: model_performance ✓')
        except Exception as e:
            print(f'  {tag}: model_performance FAILED — {e}')

        # raster
        if not args.no_raster:
            try:
                sp = os.path.join(outdir, f'raster_cluster{clust}.png')
                PatternRaster3d(data.pattern[idx, :, :, 0, :],
                                timerange=window_range, savepath=sp)
                plt.close('all')
                print(f'  {tag}: raster ✓')
            except Exception as e:
                print(f'  {tag}: raster FAILED — {e}')

    print(f'\nDone — {n_total} neurons plotted in {outdir}')

    # --- RF comparison across two segments ---
    if len(args.input) == 2:
        comp_path = os.path.join(outdir, 'rf_comparison.png')
        plot_rf_comparison(
            args.input[0], args.input[1],
            window=w, spikesorting=args.spikesorting,
            min_fr=args.min_fr, savepath=comp_path,
        )


if __name__ == '__main__':
    main()
