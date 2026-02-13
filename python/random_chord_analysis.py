#!/usr/bin/env python3
'''
Random chord spike-triggered average (STA) analysis.

Ported from DoRandomChordAnalysis.m (MATLAB).
Computes spectrotemporal receptive fields from random chord stimuli.

Authors: Brian R Mullen
Date: 2026-02-11
'''

import numpy as np
from scipy.stats import binom


def stacalc(asdf, ttls, nreps, picked_tones, tn, seg_start):
	"""
	Core STA computation for random chord stimuli.

	Equivalent to MATLAB's nested stacalc_sub function.

	Parameters
	----------
	asdf : np.ndarray (dtype=object)
		Array where each element contains spike times (ms) for one neuron.
	ttls : np.ndarray
		TTL times for this segment (absolute times in ms).
	nreps : int
		Number of stimulus repetitions.
	picked_tones : np.ndarray (dtype=object)
		(patnum, 2) array of tone index arrays from stimulus .mat file.
		MATLAB 1-indexed tone indices are converted to 0-indexed internally.
	tn : int
		Number of tone frequencies.
	seg_start : float
		Segment start time in ms (absolute).

	Returns
	-------
	stas : list
		Spike-triggered averages per neuron, each shape (tn, 10, 1, 2).
	stasigs : list
		Significance values per neuron (binomial CDF), each shape (tn, 10, 1, 2).
	nSpikes : np.ndarray
		Total spikes per neuron, shape (nNeu,).
	meanpic : np.ndarray
		Mean stimulus picture, shape (tn, 10, 1, 2).
	"""
	nNeu = len(asdf)
	patnum = len(ttls) // nreps

	# Repetition init times (segment-relative)
	inittimes = ttls[0::patnum] - seg_start
	# End boundary for the last repetition
	inittimes = np.append(inittimes, ttls[-1] + 25 - seg_start)

	# Per-repetition TTL arrays (relative to each repetition start)
	pat_ttls = []
	for i in range(nreps):
		offset = patnum * i
		pat_ttls.append(ttls[offset:offset + patnum] - seg_start - inittimes[i])

	# Filter spikes into segment-relative times
	seg_end = seg_start + inittimes[-1]
	asdf_seg = []
	for spk in asdf:
		spk = np.asarray(spk, dtype=float)
		mask = (spk >= seg_start) & (spk < seg_end)
		asdf_seg.append(spk[mask] - seg_start)

	# Split into per-repetition chunks (repetition-relative)
	asdf_minis = []
	for i in range(len(inittimes) - 1):
		mini = []
		for spk in asdf_seg:
			mask = (spk >= inittimes[i]) & (spk < inittimes[i + 1])
			mini.append(spk[mask] - inittimes[i])
		asdf_minis.append(mini)

	# Build stimulus picture matrix
	fullpic = np.zeros((tn, patnum, 2))
	for j in range(2):
		for i in range(patnum):
			tones = np.squeeze(picked_tones[i, j])
			if tones.ndim == 0:
				tones = np.array([int(tones)])
			else:
				tones = tones.astype(int)
			# Convert from 1-indexed (MATLAB) to 0-indexed (Python)
			fullpic[tones - 1, i, j] = 1

	# Pad with 0.5 values (9 columns before stimulus onset)
	picoffset = 9
	pad_before = np.ones((tn, picoffset, 2)) / 2
	pad_after = np.ones((tn, 9 - picoffset, 2)) / 2
	fullpic = np.concatenate([pad_before, fullpic, pad_after], axis=1)

	# Create sliding-window partial pictures (tn, 10, patnum, 2)
	partpic = np.zeros((tn, 10, patnum, 2))
	for i in range(patnum):
		partpic[:, :, i, :] = fullpic[:, i:i + 10, :]

	meanpic = np.mean(partpic, axis=2, keepdims=True)

	# Per-neuron STA analysis
	stas = [None] * nNeu
	stasigs = [None] * nNeu
	nSpikes = np.zeros(nNeu)

	for neu in range(nNeu):
		print('*', end='', flush=True)
		if (neu + 1) % 50 == 0:
			print(' {}'.format(neu + 1))

		# Histogram spikes into pattern bins across all repetitions
		hc = np.zeros(patnum)
		for i in range(nreps):
			edges = np.append(pat_ttls[i], pat_ttls[i][-1] + 25)
			hc += np.histogram(asdf_minis[i][neu], bins=edges)[0]

		# Find non-zero bins and compute weighted average
		fhc = np.where(hc > 0)[0]
		fhcv = hc[fhc]
		nSpike = np.sum(fhcv)
		nSpikes[neu] = nSpike

		if nSpike > 0:
			# weights shape: (1, 1, len(fhc), 1) for broadcasting
			weights = fhcv[np.newaxis, np.newaxis, :, np.newaxis]
			staraw = np.sum(partpic[:, :, fhc, :] * weights, axis=2, keepdims=True)
			stas[neu] = staraw / nSpike

			# Significance via binomial CDF
			stasigs[neu] = binom.cdf(staraw, nSpike, meanpic)
		else:
			stas[neu] = np.full((tn, 10, 1, 2), np.nan)
			stasigs[neu] = np.full((tn, 10, 1, 2), np.nan)

	print(' done!')
	return stas, stasigs, nSpikes, meanpic


def DoRandomChordAnalysis(asdf, ttls, nreps, picked_tones, tn, seg_start, nNeu):
	"""
	Run random chord STA analysis.

	Parameters
	----------
	asdf : np.ndarray (dtype=object)
		ASDF array of spike times for each neuron.
	ttls : np.ndarray
		TTL times for the segment (absolute times in ms).
	nreps : int
		Number of stimulus repetitions.
	picked_tones : np.ndarray (dtype=object)
		Tone patterns from stimulus .mat file.
	tn : int
		Number of tone frequencies.
	seg_start : float
		Segment start time in ms.
	nNeu : int
		Number of neurons.

	Returns
	-------
	dict
		'stas' : list of np.ndarray
			Spike-triggered averages per neuron, each shape (tn, 10, 1, 2).
		'stasigs' : list of np.ndarray
			Binomial CDF significance values per neuron, each shape (tn, 10, 1, 2).
		'nSpikes' : np.ndarray
			Total spike count per neuron, shape (nNeu,).
		'meanpic' : np.ndarray
			Mean stimulus picture across all patterns, shape (tn, 10, 1, 2).
	"""
	stas, stasigs, nSpikes, meanpic = stacalc(
		asdf, ttls, nreps, picked_tones, tn, seg_start
	)
	return {
		'stas': stas,
		'stasigs': stasigs,
		'nSpikes': nSpikes,
		'meanpic': meanpic,
	}
