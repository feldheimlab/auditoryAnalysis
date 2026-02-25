import os
import math

from .preprocessing import PatternToCount, probeMap

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .distributions_fit import gaussian_fit, chiSquaredTest


class load_RF_data():
	@staticmethod
	def _load_single(dataloc, spikesorting):
		'''Load data from a single directory, returning a dict of all arrays.'''
		assert os.path.exists(dataloc), 'Data directory not found: ' + dataloc
		print('Loading data from: \n\t', dataloc)

		if spikesorting == 'kilosort4':
			kilosort_loc = os.path.join(os.path.dirname(os.path.dirname(dataloc)), 'kilosort4')
			assert os.path.exists(kilosort_loc), 'Could not find kilosort data.  Expected locatation: \n\t' + kilosort_loc
			print('\nKilosort data found at the expected locatation: \n\t' + kilosort_loc)

		image_save_loc = os.path.join(os.path.dirname(dataloc), 'images')
		if not os.path.exists:
			os.mkdir(image_save_loc)
		assert os.path.exists(image_save_loc), 'Could not find image save location.  Expected locatation: \n\t' + image_save_loc
		print('\nImage save location found at the expected locatation: \n\t' + image_save_loc)

		result = {}
		result['kilosort_loc'] = kilosort_loc
		result['dataloc'] = dataloc
		result['parenetdir'] = os.path.dirname(dataloc)
		result['image_save_loc'] = image_save_loc

		window_subdir = []
		file_sublist = []
		directory_list = os.listdir(dataloc)
		for item in directory_list:
			itemq = os.path.join(dataloc, item)
			if os.path.isdir(itemq): window_subdir.append(itemq)
			if os.path.isfile(item): file_sublist.append(item)
		result['window_subdir'] = window_subdir
		result['file_list'] = file_sublist

		result['pattern'] = np.load(os.path.join(dataloc, 'pattern.npy'), allow_pickle=True)
		result['windows'] = np.load(os.path.join(dataloc, 'windows.npy'))
		result['spont_win'] = np.load(os.path.join(dataloc, 'spont_win.npy'))
		result['good_neurons'] = np.load(os.path.join(dataloc, 'good_neurons.npy'))
		result['activity_df'] = pd.read_csv(os.path.join(dataloc, 'activity-df.csv'), sep=',', index_col=0)
		
		if spikesorting == 'kilosort4':
			result['cluster'] = pd.read_csv(os.path.join(kilosort_loc, 'cluster_info.tsv'), sep='\t', index_col=0)
			result['templates'] = np.load(os.path.join(kilosort_loc, 'templates.npy'))
			result['channelposition'] = np.load(os.path.join(kilosort_loc, 'channel_positions.npy'))
		elif spikesorting == 'vision':
			import scipy.io as sio
			result['templates'] = sio.loadmat(os.path.join(result['parenetdir'], 'eisummary.mat'))
			result['channelposition'] = np.flip(probeMap(probe='AN'), 1)

		result['aic_bic'] = []
		result['datafits'] = []
		result['modelfits'] = []
		result['neuron_assess'] = []
		result['parameters'] = []
		result['sumresid'] = []
		result['variances'] = []

		for w, window in enumerate(result['windows']):
			for subdir in window_subdir:
				if subdir.endswith(str(w)):
					current_subidir = subdir
					break
			print('\nLoading fit data from {0} Window, {1} ms:  {2}'.format(w, window,
				  os.path.basename(os.path.normpath(current_subidir))))
			result['aic_bic'].append(np.load(os.path.join(current_subidir, 'aic_bic.npy')))
			result['datafits'].append(np.load(os.path.join(current_subidir, 'datafits.npy')))
			result['modelfits'].append(np.load(os.path.join(current_subidir, 'modelfits.npy')))
			result['neuron_assess'].append(np.load(os.path.join(current_subidir, 'neuron_assess.npy')))
			result['parameters'].append(np.load(os.path.join(current_subidir, 'parameters.npy')))
			result['sumresid'].append(np.load(os.path.join(current_subidir, 'sumresid.npy')))
			result['variances'].append(np.load(os.path.join(current_subidir, 'variances.npy')))

		# Per-window RF classification: Kent BIC < uniform BIC
		# aic_bic columns: [uniform_aic, kent_aic, uniform_bic, kent_bic]
		result['is_rf'] = []
		result['rf_indices'] = []
		result['delta_bic'] = []
		for w in range(len(result['windows'])):
			ab = result['aic_bic'][w]
			bic_uniform = ab[:, 2]
			bic_kent = ab[:, 3]
			dbic = bic_uniform - bic_kent
			mask = dbic > 0
			result['is_rf'].append(mask)
			result['rf_indices'].append(np.where(mask)[0])
			result['delta_bic'].append(dbic)
			print(f'  Window {w}: {mask.sum()} / {len(mask)} RF neurons')

		return result

	def __init__(self, dataloc, spikesorting='kilosort4'):
		'''Load receptive field analysis results from the pipeline output directory.

		Reads pattern arrays, analysis windows, activity DataFrames, Kent
		distribution fit results, and cluster information produced by the
		auditory analysis pipeline.

		Parameters
		----------
		dataloc : str or list of str
			Path to a segment results directory, e.g.
			``'results/seg5 mult 10/'``, or a list of such paths to
			concatenate data across multiple recording sessions.
		spikesorting : str, optional
			Spike sorter used to produce the data. Must be ``'kilosort4'``
			or ``'vision'``. Default is ``'kilosort4'``.

		Attributes
		----------
		pattern : np.ndarray
			5-D spike pattern array loaded from ``pattern.npy``.
		windows : np.ndarray
			Analysis time windows loaded from ``windows.npy``.
		activity_df : pd.DataFrame
			Per-neuron activity metrics loaded from ``activity-df.csv``.
		good_neurons : np.ndarray
			Indices of neurons passing quality criteria.
		aic_bic : list of np.ndarray
			AIC/BIC values for each analysis window.
		datafits : list of np.ndarray
			Observed firing-rate maps for each window.
		modelfits : list of np.ndarray
			Model-predicted firing-rate maps for each window.
		parameters : list of np.ndarray
			Fitted distribution parameters for each window.
		variances : list of np.ndarray
			Parameter variance estimates for each window.
		neuron_assess : list of np.ndarray
			Per-neuron fit assessment flags for each window.
		templates : np.ndarray
			Spike waveform templates (kilosort4) or EI summary (vision).
		channelposition : np.ndarray
			XY positions of recording channels on the probe.
		source_labels : list of str
			Basename of each source directory.
		source_indices : np.ndarray
			Index (0, 1, ...) mapping each neuron to its source directory.
		'''
		if isinstance(dataloc, str):
			dataloc = [dataloc]

		datasets = [self._load_single(d, spikesorting) for d in dataloc]

		self.spikesorting = spikesorting
		self.azimuths = np.arange(17)*18 - 144
		self.elevations = np.arange(5)*20

		# Use first directory for path-based attributes
		self.dataloc = dataloc
		self.kilosort_loc = datasets[0]['kilosort_loc']
		self.parenetdir = datasets[0]['parenetdir']
		self.image_save_loc = datasets[0]['image_save_loc']
		self.window_subdir = datasets[0]['window_subdir']
		self.file_list = datasets[0]['file_list']
		print('\nImages will be saved at: \n\t', self.image_save_loc)

		self.source_labels = [os.path.basename(os.path.normpath(d)) for d in dataloc]

		if len(datasets) == 1:
			# Single directory — assign directly (backward compatible)
			d = datasets[0]
			self.pattern = d['pattern']
			self.windows = d['windows']
			self.spont_win = d['spont_win']
			self.good_neurons = d['good_neurons']

			self.activity_df = d['activity_df']
			if spikesorting == 'kilosort4':
				self.cluster = d['cluster']
			self.templates = d['templates']
			if 'channelposition' in d:
				self.channelposition = d['channelposition']
			self.aic_bic = d['aic_bic']
			self.datafits = d['datafits']
			self.modelfits = d['modelfits']
			self.neuron_assess = d['neuron_assess']
			self.parameters = d['parameters']
			self.sumresid = d['sumresid']
			self.variances = d['variances']
			self.source_indices = np.zeros(len(self.good_neurons), dtype=int)
			self.is_rf = d['is_rf']
			self.rf_indices = d['rf_indices']
			self.delta_bic = d['delta_bic']
		else:
			# Multiple directories — concatenate along neuron axis
			n_windows_list = [len(d['windows']) for d in datasets]
			assert all(n == n_windows_list[0] for n in n_windows_list), \
				'All directories must have the same number of analysis windows. Got: ' + str(n_windows_list)

			self.windows = datasets[0]['windows']
			self.spont_win = datasets[0]['spont_win']

			# Concatenate pattern along neuron axis (axis 0)
			self.pattern = np.concatenate([d['pattern'] for d in datasets], axis=0)

			# Offset good_neurons cluster IDs to prevent collisions
			offset_neurons = []
			offset = 0
			source_indices = []
			for i, d in enumerate(datasets):
				offset_neurons.append(d['good_neurons'] + offset)
				source_indices.append(np.full(len(d['good_neurons']), i, dtype=int))
				if len(d['good_neurons']) > 0:
					offset = d['good_neurons'].max() + 1 + offset
			self.good_neurons = np.concatenate(offset_neurons)
			self.source_indices = np.concatenate(source_indices)

			# Activity DataFrame
			self.activity_df = pd.concat([d['activity_df'] for d in datasets], ignore_index=True)

			# Cluster info (kilosort4)
			if spikesorting == 'kilosort4':
				cluster_frames = []
				idx_offset = 0
				for i, d in enumerate(datasets):
					c = d['cluster'].copy()
					c.index = c.index + (offset_neurons[i][0] - datasets[i]['good_neurons'][0]) if len(datasets[i]['good_neurons']) > 0 else c.index
					cluster_frames.append(c)
				self.cluster = pd.concat(cluster_frames)

			# Templates
			self.templates = np.concatenate([d['templates'] for d in datasets], axis=0)

			# Channel positions — use first directory (same probe assumed)
			if 'channelposition' in datasets[0]:
				self.channelposition = datasets[0]['channelposition']

			# Per-window arrays
			self.aic_bic = []
			self.datafits = []
			self.modelfits = []
			self.neuron_assess = []
			self.parameters = []
			self.sumresid = []
			self.variances = []
			self.is_rf = []
			self.rf_indices = []
			self.delta_bic = []

			for w in range(n_windows_list[0]):
				self.aic_bic.append(np.concatenate([d['aic_bic'][w] for d in datasets], axis=0))
				self.datafits.append(np.concatenate([d['datafits'][w] for d in datasets], axis=0))
				self.modelfits.append(np.concatenate([d['modelfits'][w] for d in datasets], axis=0))
				self.neuron_assess.append(np.concatenate([d['neuron_assess'][w] for d in datasets], axis=0))
				self.parameters.append(np.concatenate([d['parameters'][w] for d in datasets], axis=0))
				self.sumresid.append(np.concatenate([d['sumresid'][w] for d in datasets], axis=0))
				self.variances.append(np.concatenate([d['variances'][w] for d in datasets], axis=0))
				self.is_rf.append(np.concatenate([d['is_rf'][w] for d in datasets]))
				self.rf_indices.append(np.where(self.is_rf[-1])[0])
				self.delta_bic.append(np.concatenate([d['delta_bic'][w] for d in datasets]))

	def rf_neurons(self, window=0):
		'''Identify neurons with receptive fields using BIC model comparison.

		A neuron is classified as having a receptive field when the Kent
		distribution BIC is lower (better fit) than the uniform distribution
		BIC, indicating spatially tuned responses.

		Parameters
		----------
		window : int, optional
			Analysis time window index. Default is ``0``.

		Returns
		-------
		rf_indices : np.ndarray
			Indices into the neuron axis where the Kent model is preferred.
		'''
		# aic_bic columns: [uniform_aic, kent_aic, uniform_bic, kent_bic]
		ab = self.aic_bic[window]
		bic_uniform = ab[:, 2]
		bic_kent = ab[:, 3]
		
		delta_bic = bic_uniform - bic_kent

		return np.where(delta_bic > 0)[0]


def normalize_templates(template_matrix, norm_factor=None):

	"""
	Returns the normalized template matrix.  
	If norm_factor is None, it will calculate the norm_factor as the max value of each template

	Arguments:
		template_matrix: 3D array of the mean templates (n waveforms x n timepoints x n channels)
		norm_factor: 2D array to normalize against (n timepoints x n waveforms)

	Returns:
		templates_norm: 3D array of the normalized templates (n waveforms x n timepoints x n channels) 
		norm_factor: 2D array to normalize against (n timepoints x n waveforms)

	"""
	templates_norm = np.swapaxes(template_matrix, 0, 2)
	try:
		norm_factor.shape
	except Exception as e:
		norm_factor = np.nanmax(np.abs(templates_norm), axis=(0,1))
	templates_norm = np.divide(templates_norm, norm_factor, out=np.zeros_like(templates_norm), 
							   where=norm_factor!=0)
	templates_norm = np.swapaxes(templates_norm, 0, 2)

	return templates_norm, norm_factor


def plot_neurons_relative_to_probe(data_obj, save_image_dir):
		'''Plot neuron cluster locations relative to the recording probe layout.

		Generates a figure showing where each neuron cluster sits on the
		physical probe geometry, colour-coded by classification group, and
		saves the result as a PNG image.

		Parameters
		----------
		data_obj : data_load
			Loaded data object containing ``chanposition``, ``cluster``,
			``class_col``, and ``spikesorting`` attributes.
		save_image_dir : str
			Directory path where the output image will be saved.
		'''
		if data_obj.spikesorting == 'vision':
			fig, axs = plt.subplots(1,2, figsize = (5,5), sharey=True)    

			axs[0].scatter(data_obj.chanposition[:,0],data_obj.chanposition[:,1], facecolors='None', edgecolors='k')
			axs[0].scatter(data_obj.basicinfo['y'],data_obj.basicinfo['x'], c='purple', alpha=0.25, vmin=0, vmax=2)
			axs[0].set_ylim(-50,1150)
			axs[0].set_ylabel('distance from tip (um)')
			axs[1].set_xlabel('span (um)')
			axs[1].set_xlabel('FR')
			print(fr.shape, np.arange(185))
			im1 = axs[1].scatter(fr, np.arange(183),  c='purple', alpha = 0.25)
			plt.savefig(os.path.join(save_image_dir, 'cluster_loc_wnoise.png'), dpi=300)

		if data_obj.spikesorting == 'kilosort4':
			data_obj.cluster['group_c'] = 0

			groups = np.unique(data_obj.cluster[data_obj.class_col])
			for i, group in enumerate(groups):
				data_obj.cluster.loc[data_obj.cluster[data_obj.cluster[data_obj.class_col]==group].index, 'group_c'] = int(i)

			fig, axs = plt.subplots(1,len(groups)+1, figsize = (10,5), sharey=True)    
			for j, group in enumerate(groups):
				axs[j].scatter(data_obj.chanposition[:,0], data_obj.chanposition[:,1], facecolors='None', edgecolors='k')
				clust = data_obj.cluster[data_obj.cluster[data_obj.class_col]==group].copy()
				for index in clust.index:
					pos = data_obj.chanposition[clust.loc[index, 'ch']]
					color = clust.loc[index, 'group_c']
					axs[j].scatter(pos[0], pos[1], c=color, alpha=0.25, vmin=0, vmax=2)
					axs[j].set_ylim(-50,800)
				axs[j].set_title(group)
					
			axs[0].set_ylabel('distanec from tip (um)')
			axs[1].set_xlabel('span (um)')
			axs[-1].set_xlabel('FR')

			im1 = axs[-1].scatter(data_obj.cluster['fr'], data_obj.cluster['depth'],  c=data_obj.cluster['group_c'], alpha = 0.25, vmax = 2)
			divider = make_axes_locatable(axs[-1])

			cax = divider.append_axes('right', size='5%', pad=0.05)
			cbar = fig.colorbar(im1, cax =cax , orientation='vertical')
			cbar.set_ticks(np.arange(len(groups)))
			cbar.set_ticklabels(groups)
			plt.savefig(os.path.join(save_image_dir, 'cluster_loc_relative_to_probe.png'), dpi=300)


def PatternRaster3d(pattern3d, timerange=None, savepath=None):
	'''Create and save a raster plot from a 3D spike pattern array.

	Each sub-panel corresponds to one elevation/azimuth combination, with
	individual trials stacked vertically and spike times plotted along the
	horizontal axis.

	Parameters
	----------
	pattern3d : nested list or array
		Three-level nested structure indexed as
		``[elevation][azimuth][trial]``, where each innermost element is
		an array of spike times.
	timerange : list or None, optional
		Time axis limits in milliseconds, given as ``[start, end]``. A
		single-element list or scalar sets the upper bound with a lower
		bound of 0. Default is ``None`` (uses 0--1000 ms).
	savepath : str or None, optional
		File path to save the figure. Default is ``None``.
	'''
	ns = np.zeros(3)
	ns[0] = len(pattern3d)
	ns[1] = len(pattern3d[0])
	ns[2] = len(pattern3d[0][0])
	
	print('Raster plot creation of size: ', ns)
	if timerange is None:
		maxtime = 1000
		mintime = 0
	elif not isinstance(timerange, list):
		mintime = 0
		maxtime = timerange
	elif len(timerange) == 1:
		mintime = 0
		maxtime = timerange[0]
	elif len(timerange)==2:
		mintime = timerange[0]
		maxtime = timerange[1]
		
	dur = maxtime - mintime
	
	fig, axs = plt.subplots(int(ns[0]), int(ns[1]), figsize=(10,2.5))
	
	xpos = int(0)
	x = int(0)
	if ns[0]!=1:
		for x in np.arange(ns[0]):
			x = int(ns[0]-1-x)
			for y in np.arange(ns[1]):
				y = int(y)
				for t in np.arange(ns[2]):
					t = int(t)
					try:
						xs = np.squeeze(pattern3d[x][y][t])
		#                 print(xs)
						try:
							ys = np.ones(xs.shape)*t
						except:
							ys = np.ones(xs.shape[0])*t
					except Exception as e:
						print(e)
						xs = np.nan
						ys = np.nan
					if np.isnan(xs).any(): continue
					axs[x][y].scatter(xs, ys, s=1, marker='|', linewidth=0.5, c='k')
					xs = np.nan
				axs[x][y].set_yticks([])
				axs[x][y].set_xticks([])
				axs[x][y].set_xlim([mintime, maxtime])
				axs[x][y].set_ylim([-1, ns[2]+1])
		axs[int(ns[0]-1)][0].set_yticks([0, ns[2]])
		axs[int(ns[0]-1)][0].set_xticks([mintime, maxtime])
		axs[int(ns[0]-1)][0].set_xticklabels([0, dur])
		axs[int(ns[0]-1)][0].set_xlabel('time (ms)')
		axs[int(ns[0]-1)][0].set_ylabel('trial')
		
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.savefig(savepath, dpi=300)
		plt.show()
	else:
		for y in np.arange(ns[1]):
			y = int(y)
			for t in np.arange(ns[2]):
				t = int(t)
				try:
					xs = np.squeeze(pattern3d[0][y][t])
	#                 print(xs)
					try:
						ys = np.ones(xs.shape)*t
					except:
						ys = np.ones(xs.shape[0])*t
				except Exception as e:
					print(e)
					xs = np.nan
					ys = np.nan
				if np.isnan(xs).any(): continue
				axs[y].scatter(xs, ys, s=1, c='k')
				xs = np.nan
			axs[y].set_yticks([])
			axs[y].set_xticks([])
			axs[y].set_xlim([mintime, maxtime])
			axs[y].set_ylim([-1, ns[2]+1])
		axs[0].set_yticks([0, ns[2]])
		axs[0].set_xticks([mintime, maxtime])
		axs[0].set_xticklabels([0, dur])
		axs[0].set_xlabel('time (ms)')
		axs[0].set_ylabel('trial')
		plt.subplots_adjust(wspace=0, hspace=0)
		plt.savefig(savepath, dpi=300)
		plt.show()


def PSTH(data_obj, index, timerange, timeBinSz):
	'''Compute a peri-stimulus time histogram (PSTH).

	Sums binned spike counts across all stimulus conditions (elevation,
	azimuth, repetition) for a single neuron to produce an aggregate
	temporal response profile.

	Parameters
	----------
	data_obj : load_RF_data
		Loaded receptive-field data object whose ``pattern`` attribute
		contains the 5-D spike pattern array.
	index : int
		Neuron index into ``data_obj.pattern``.
	timerange : list
		Two-element list ``[start, end]`` specifying the time window in
		milliseconds.
	timeBinSz : int
		Histogram bin size in milliseconds.

	Returns
	-------
	histogram : np.ndarray
		Summed spike counts per time bin across all conditions.
	bins : np.ndarray
		Bin edges returned by the underlying histogram computation.
	'''
	fr, bins = PatternToCount(data_obj.pattern[index], timerange=timerange, timeBinSz = timeBinSz, verbose=True)
	histogram = np.sum(fr, axis=(0,1,2,3))
	
	return histogram, bins


def model_performance(data_obj, index=None, savepath=None):
	'''Plot side-by-side comparison of observed data and Kent model fits.

	For each analysis time window, displays the observed mean firing-rate
	map (top row) and the corresponding Kent distribution model prediction
	(bottom row) as heat-map images.

	Parameters
	----------
	data_obj : load_RF_data
		Loaded receptive-field data object containing ``datafits``,
		``modelfits``, ``windows``, ``elevations``, and ``azimuths``.
	index : int or None, optional
		Neuron index to plot. Default is ``None``.
	savepath : str or None, optional
		File path to save the figure. If ``None``, the figure is shown
		but not saved. Default is ``None``.
	'''
	n_windows = len(data_obj.windows)
	fig, axs = plt.subplots(2,n_windows, figsize=(n_windows*2,2.5))
	for w, window in enumerate(data_obj.windows):
		vmax = np.max(data_obj.datafits[w][index])
		vmin = 0
		if n_windows > 1:
			axs[0][w].imshow(data_obj.datafits[w][index], vmin=vmin, vmax=vmax)
			axs[1][w].imshow(data_obj.modelfits[w][index], vmin=vmin, vmax=vmax)
			axs[0][w].set_yticks([0,2,4])
			axs[0][w].set_xticks([0,4,8,12,16])
			axs[1][w].set_yticks([0,2,4])
			axs[1][w].set_xticks([0,4,8,12,16])
			axs[0][w].set_ylim([-0.5, 4.5])
			axs[1][w].set_ylim([-0.5, 4.5])
			if w != 0:
				axs[0][w].set_yticklabels([])
				axs[1][w].set_yticklabels([])
			else:
				axs[0][w].set_yticklabels(data_obj.elevations[[0,2,4]])
				axs[1][w].set_yticklabels(data_obj.elevations[[0,2,4]])
			axs[0][w].set_xticklabels([])
			axs[1][w].set_xticklabels(data_obj.azimuths[[0,4,8,12,16]])
			axs[0][w].set_title(window)
		else:
			axs[0].imshow(data_obj.datafits[w][index], vmin=vmin, vmax=vmax)
			axs[1].imshow(data_obj.modelfits[w][index], vmin=vmin, vmax=vmax)
			axs[0].set_yticks([0,2,4])
			axs[0].set_xticks([0,4,8,12,16])
			axs[1].set_yticks([0,2,4])
			axs[1].set_xticks([0,4,8,12,16])
			axs[0].set_xticklabels(data_obj.azimuths[[0,4,8,12,16]])
			axs[0].set_yticklabels(data_obj.elevations[[0,2,4]])
			axs[1].set_yticklabels(data_obj.elevations[[0,2,4]])
			axs[0].set_title(window)
			axs[0].set_ylim([-0.5, 4.5])
			axs[1].set_ylim([-0.5, 4.5])
	if n_windows > 1:
		axs[0][0].set_ylabel('Avg (elev.)')
		axs[1][0].set_ylabel('Model (elev.)')
		axs[1][1].set_xlabel('azimuth')
	
	plt.tight_layout() 
	if savepath != None:
		plt.savefig(savepath, dpi=300)
	plt.show()


def cluster_info_waveform(data_obj, cluster=None,
						  index=None, timeBinSz=1,
						  timerange=[0,20], savepath=None):
	'''Plot a neuron waveform template, PSTH with Gaussian fit, and probe location.

	Produces a three-panel figure: (1) the normalised spike waveform on the
	most active channel, (2) a peri-stimulus time histogram with an
	overlaid Gaussian fit, and (3) the neuron's position on the recording
	probe.

	Parameters
	----------
	data_obj : load_RF_data
		Loaded receptive-field data object containing templates, channel
		positions, pattern data, and spike-sorting metadata.
	cluster : int or None, optional
		Cluster ID used when ``spikesorting == 'kilosort4'`` to look up
		the waveform template and channel info. Default is ``None``.
	index : int or None, optional
		Neuron index into the pattern array (used for PSTH computation
		and for vision spike-sorting lookups). Default is ``None``.
	timeBinSz : int, optional
		PSTH histogram bin size in milliseconds. Default is ``1``.
	timerange : list, optional
		Two-element list ``[start, end]`` in milliseconds defining the
		PSTH time window. Default is ``[0, 20]``.
	savepath : str or None, optional
		File path to save the figure. Default is ``None``.
	'''
	if data_obj.spikesorting == 'kilosort4':
		group = data_obj.cluster.loc[cluster, 'group']
		print(group)
		channel = int(data_obj.cluster.loc[cluster, 'most_active_channel'])
		templates_norm,_ = normalize_templates(data_obj.templates)
		temp = templates_norm[cluster,:,:]
		waveform = temp[:,channel]
		loc = np.argmin(waveform)
		fpms = 30
		time_shift = (np.arange(61)-loc)/fpms
	elif data.spikesorting=='vision':
		fpms = 20
		channel = data_obj.templates['maxChannels'][index]
		waveform = data_obj.templates['waveforms'][index]
		loc = np.argmin(waveform)
		time_shift = (np.arange(121)-loc)/fpms
	
	fig, axs = plt.subplots(1,3, figsize=(6,3))

	axs[0].plot(time_shift, waveform, color='k')
	axs[0].set_xticks([0,1])
	axs[0].set_xlabel('time (ms)')
	axs[0].set_ylabel('Normalized Amp')
	axs[0].set_title('Waveform')
#     axs[0].set_xlim([time_shift[int(loc-20)], time_shift[int(loc+40)]])
	histogram, bins = PSTH(data_obj, index, timerange, timeBinSz)
	g_fit = gaussian_fit(histogram, bins[1:])

	axs[1].bar(bins[1:], histogram, width=timeBinSz, color='b')
	axs[1].plot(bins[1:], g_fit.predict(bins[1:]), linestyle=':', color='k')
	axs[1].set_xticks([timerange[0], np.mean(timerange), timerange[1]])
	axs[1].set_xlabel('time (ms)')
	axs[1].set_ylabel('AP count')
	axs[1].set_ylim(0, np.max(histogram)+3)
	axs[1].set_title('PSTH')
	
	ch = np.squeeze(data_obj.channelposition[channel,:])
	print(ch.shape)
	axs[2].scatter(data_obj.channelposition[:,0], data_obj.channelposition[:,1], facecolors='None', edgecolors='k')
	axs[2].scatter(ch[0], ch[1], facecolors='Red', edgecolors='k')
	axs[2].set_xlabel('AP distance(um)')
	axs[2].set_ylabel('distance from shank tip(um)')
	axs[2].set_title('Most Active Channel')

	axs[2].scatter(ch[0], ch[1], facecolors='Red', edgecolors='k')

	plt.tight_layout()
	print(savepath)
	fig.savefig(savepath, dpi=300)
	plt.show()


class TopographicMap:
	def __init__(self, data_obj, window=0):
		'''Extract cluster positions in rotated auditory space from Kent fit parameters.

		Parameters
		----------
		data_obj : load_RF_data
			Loaded receptive-field data object.
		window : int, optional
			Analysis time window index. Default is ``0``.
		'''
		params = data_obj.parameters[window]
		sem_values = data_obj.variances[window]
		cluster_ids = data_obj.good_neurons

		theta = params[:, 2]
		phi = params[:, 3]

		self.elevation = np.degrees(theta)
		self.azimuth = np.degrees(phi)
		self.elevation_var = np.degrees(np.sqrt(np.abs(sem_values[:, 2])))
		self.azimuth_var = np.degrees(np.sqrt(np.abs(sem_values[:, 3])))

		angle = np.radians(15)
		self.x = self.azimuth * np.cos(angle) - self.elevation * np.sin(angle)
		self.y = self.azimuth * np.sin(angle) + self.elevation * np.cos(angle)
		self.cluster_ids = cluster_ids
		self.data_obj = data_obj

	def plot(self, values=None, cmap='viridis', label='', savepath=None):
		'''Scatter plot of cluster positions in rotated auditory space.

		Parameters
		----------
		values : array-like or None, optional
			Color values for each cluster. If ``None``, uses firing rate
			from ``data_obj.cluster['fr']``.
		cmap : str, optional
			Matplotlib colormap name. Default is ``'viridis'``.
		label : str, optional
			Colorbar label. Default is ``''``.
		savepath : str or None, optional
			File path to save the figure. Default is ``None``.
		'''
		if values is None:
			values = self.data_obj.cluster.loc[self.cluster_ids, 'fr'].values
			if label == '':
				label = 'Firing Rate (Hz)'

		fig, ax = plt.subplots(figsize=(6, 5))
		sc = ax.scatter(self.x, self.y, c=values, cmap=cmap, edgecolors='k', linewidths=0.5)
		ax.set_xlabel('Azimuth (deg)')
		ax.set_ylabel('Elevation (deg)')
		ax.set_title('Topographic Map (Anatomical)')
		cbar = plt.colorbar(sc, ax=ax)
		cbar.set_label(label)
		plt.tight_layout()
		if savepath is not None:
			plt.savefig(savepath, dpi=300)
		plt.show()
