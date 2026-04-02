#!/usr/bin/env python3
'''
Full pipeline for analysis. All custom functions used are found in ./python

Authors: Brian R Mullen
Date: 2026-02-10
'''


import os
import math
import yaml
import json

from scipy.stats import poisson
from scipy.stats import nbinom
import scipy.io as sio

# custom functions
from python.preprocessing import probeMap, probeMapFromMeta, readStimFile, getTTLseg, patternGen, PatternToCount, sigAudFRCompareSpont
from python.distributions_fit import azimElevCoord, kent_fit, uniform_fit, aic_leastsquare, bic_leastsquare
from python.visualizations import plot_neurons_relative_to_probe, plot_overall_fr_on_probe, plot_windowed_fr_on_probe
from python.random_chord_analysis import DoRandomChordAnalysis
from config import configs
from python.hdf5manager import hdf5manager as h5

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool, cpu_count


def _fit_neuron_kent(args_tuple):
	"""Worker function for parallel Kent distribution fitting.

	Must be at module level to be picklable by multiprocessing.
	"""
	n, neuron_data, azim, elev, laser = args_tuple
	from python.distributions_fit import azimElevCoord, kent_fit, uniform_fit, aic_leastsquare, bic_leastsquare

	if laser is not None:
		results = {}
		for l in laser:
			if l == 0:
				mean_data = np.mean(np.squeeze(neuron_data[:,:,0]), axis=2)
			else:
				mean_data = np.mean(np.squeeze(neuron_data[:,:,1]), axis=2)

			xs = azimElevCoord(azim, elev, np.mean(np.squeeze(neuron_data), axis=2))
			xyz = xs[:,:3]
			aud = xs[:,3]
			kentfit = kent_fit(aud, xyz, datashape=mean_data.shape, verbose=False)
			uniformfit = uniform_fit(aud, np.arange(mean_data.size))

			uaic = aic_leastsquare(uniformfit.residuals, uniformfit.params)
			kaic = aic_leastsquare(kentfit.residuals, kentfit.params)
			ubic = bic_leastsquare(uniformfit.residuals, uniformfit.params)
			kbic = bic_leastsquare(kentfit.residuals, kentfit.params)

			results[l] = {
				'params': kentfit.params, 'var': kentfit.var,
				'fitdist': kentfit.fitdist, 'data': kentfit.data,
				'aic_bic': [uaic, kaic, ubic, kbic],
				'sumresid': [uniformfit.residuals_sum, kentfit.residual_sum]
			}
		return n, results
	else:
		mean_data = np.mean(np.squeeze(neuron_data), axis=2)

		xs = azimElevCoord(azim, elev, np.mean(np.squeeze(neuron_data), axis=2))
		xyz = xs[:,:3]
		aud = xs[:,3]
		kentfit = kent_fit(aud, xyz, datashape=mean_data.shape, verbose=False)
		uniformfit = uniform_fit(aud, np.arange(mean_data.size))

		uaic = aic_leastsquare(uniformfit.residuals, uniformfit.params)
		kaic = aic_leastsquare(kentfit.residuals, kentfit.params)
		ubic = bic_leastsquare(uniformfit.residuals, uniformfit.params)
		kbic = bic_leastsquare(kentfit.residuals, kentfit.params)

		return n, {
			'params': kentfit.params, 'var': kentfit.var,
			'fitdist': kentfit.fitdist, 'data': kentfit.data,
			'aic_bic': [uaic, kaic, ubic, kbic],
			'sumresid': [uniformfit.residuals_sum, kentfit.residual_sum]
		}


def get_IDs(cluster: pd.DataFrame, 
			class_col: str, 
			group: str = 'good'):
	'''
	Gets the cluster of IDs of the classification indicated
	
	Arguments:
		cluster: pd.DataFrame the indicates classification of clusters (usually saved from phy)
		class_col: column in the pd.DataFrame that the classification is in
		group: which classification used, could be 'good', 'mua', 'noise'

	Returns:
		IDs: cluster ids from sorting to be included in the saved output

	'''
	if group=='all':
		IDs = cluster['cluster_id'].values
		IDs_index = cluster['cluster_id'].index
	else:
		IDs = cluster.loc[cluster[class_col]==group, 'cluster_id'].values
		IDs_index = cluster.loc[cluster[class_col]==group, 'cluster_id'].index

	return np.array(IDs, dtype=int),  np.array(IDs_index, dtype=int)


class data_load():
	'''
	Load spike-sorted data into a standardized format for pipeline processing.

	Supports kilosort4 (.npy/.tsv) and VISION (.mat) spike sorting outputs.
	Converts raw spike data into ASDF format (array of spike time lists per neuron)
	and loads associated metadata (TTL times, channel maps, data segment boundaries).

	Attributes
	----------
	spikesorting : str
		Spike sorter used ('kilosort4' or 'vision').
	asdf : np.ndarray (dtype=object)
		Array where each element is a list/array of spike times (ms) for one neuron.
	ttl_times : np.ndarray
		TTL pulse times in ms from the recording.
	datasep : np.ndarray
		Timestamps marking boundaries between data segments.
	IDs : np.ndarray
		Cluster IDs of neurons included in the analysis.
	chanmap : np.ndarray
		Channel map of the recording probe.
	chanposition : np.ndarray
		(n_channels, 2) array of channel x,y positions on the probe.
	'''
	def __init__(self, dataloc, spikesorting, class_col, group, rate, f):
		'''
		Parameters
		----------
		dataloc : str
			Path to the spike-sorted output directory.
		spikesorting : str
			Spike sorter used ('kilosort4' or 'vision').
		class_col : str
			Column name in cluster TSV for neuron classification.
		group : str
			Which classification to include ('good', 'mua', or 'all').
		rate : int
			Sampling rate in Hz (e.g. 30000).
		f : file object
			Open file handle for logging output.
		'''
		self.spikesorting = spikesorting
		parentdir = os.path.dirname(dataloc)
		self.class_col = class_col
		self.group = group
		self.rate = rate

		if self.spikesorting  == 'vision':
			#VISION SPECIFIC
			asdfpath = os.path.join(dataloc, 'asdf.mat')
			risedpath = os.path.join(dataloc, 'ttlTimes.mat')
			basicinfopath = os.path.join(dataloc, 'xy.mat')
			segmentlegths = os.path.join(dataloc, 'segmentlengths.mat')
			segmentstims = os.path.join(dataloc, 'segmentstims.mat')

			segment = os.path.join(dataloc, 'segttls.mat')

			basicinfo = sio.loadmat(basicinfopath)#.load()
			self.ttl_times = np.squeeze(sio.loadmat(risedpath)['ttlTimes'])
			segttls = sio.loadmat(segment)#.load()
			# segmentstim = sio.loadmat(segmentstims)#.load()

			datasep = np.squeeze(sio.loadmat(segmentlegths)['segmentseparations'])
			datasep2 = np.zeros(datasep.shape[0]+1)
			datasep2[1:] = datasep
			self.datasep = datasep2

			#load data created by kilosort
			self.chanmap = probeMap(probe='AN') #map of the probe
			self.chanposition = self.chanmap# np.load(os.path.join(kilosortloc,'channel_positions.npy'))


			asdf_raw = sio.loadmat(asdfpath)['asdf_raw']
			# aud_ids = sio.loadmat(os.path.join(dataloc, 'AuditorySpotSummary_1-posneu.mat'))['posneu']
			print(asdf_raw.shape)
			nneur = 0 
			nextn = True
			totalAPs = 0 
			while nextn:
			    try:
			        totalAPs += np.squeeze(asdf_raw[nneur][0]).shape[0]
			        nneur += 1
			    except:
			        nextn = False
			        
			#simplify the asdf and determine fr
			self.asdf = np.empty((nneur), dtype='object')
			self.fr = np.empty(nneur)
			self.IDs = np.arange(nneur)

			for n in np.arange(nneur):
			    self.asdf[n] = np.squeeze(asdf_raw[n][0])
			    self.fr[n] = len(self.asdf[n])/datasep[-1]

		elif self.spikesorting  == 'kilosort4':
			#spikes
			spiketemplates = np.load(os.path.join(dataloc, 'spike_clusters.npy')) # to make asdf
			spiketimes = np.load(os.path.join(dataloc, 'spike_times.npy')) # to make asdf
			self.templates = np.load(os.path.join(dataloc, 'templates.npy')) #waveforms

			try:
				cluster = pd.read_csv(os.path.join(dataloc, 'cluster_info.tsv'), sep='\t') #class 
			except:
				cluster = pd.read_csv(os.path.join(dataloc, 'cluster_group.tsv'), sep='\t')
			
			cluster.dropna(subset=[self.class_col], inplace=True)
			keep_cols = ['cluster_id', self.class_col, 'ch', 'depth', 'fr']
			if 'most_active_channel' in cluster.columns:
				keep_cols.append('most_active_channel')
			self.cluster = cluster[keep_cols]
			
			groups = np.unique(cluster[class_col])

			print('Breakdown of kilosort4 outcome: ', file=f)
			for group in groups:
				frac = len(cluster[cluster[class_col]==group])/len(cluster)
				print('\t{0} fraction {1}/{2}: {3}%'.format(group, len(cluster[cluster[class_col]==group]), 
														 len(cluster), np.round(frac*100, 2)), file=f)
			

			#created from rhd files Save the ttl times in a file so we don't have to determine this every time
			# digital = np.load(os.path.join(kilosortloc, 'digital.npy')) #get ttls
			try:
				datasep = np.load(os.path.join(dataloc, 'datasep.npy'), allow_pickle=True).item() #get data separations 
			except:
				datasep = np.load(os.path.join(parentdir, 'datasep.npy'), allow_pickle=True).item()
			
			self.datasep = datasep['Datasep']
			
			self.IDs, IDs_index = get_IDs(self.cluster, class_col=self.class_col, group=self.group )
			print('\nProcessing {} neurons: '.format(self.group), file=f)
			print('\nCluster description: ', file=f)
			print(cluster.describe(), file=f)

			#load channel maps
			self.chanmap = np.load(os.path.join(dataloc,'channel_map.npy')) #map of the probe
			# Try to extract probe map from SpikeGLX .meta file, fall back to hardcoded probeMap
			import glob
			meta_files = glob.glob(os.path.join(parentdir, '*.ap.meta'))
			if meta_files:
				self.chanposition = probeMapFromMeta(meta_files[0])
				print('Probe map extracted from meta file: {}'.format(meta_files[0]), file=f)
			else:
				self.chanposition = np.load(os.path.join(dataloc,'channel_positions.npy'))
			try:
				self.ttl_times = np.load(os.path.join(dataloc, 'ttlTimes.npy'))
			except: 
				self.ttl_times= np.load(os.path.join(parentdir, 'ttlTimes.npy'))

			neuron_clust = np.unique(spiketemplates)
			max_clust = len(self.IDs)
			self.asdf = np.empty((max_clust), dtype=object)

			for n, clust in enumerate(self.IDs):
				try:
					self.asdf[n]=list(np.squeeze(spiketimes[spiketemplates==clust])/(self.rate/1000))
				except:
					print(n)
					self.asdf[n]=[]


if __name__ == '__main__':

	import argparse
	import time
	import datetime

	# Argument Parsing
	# -----------------------------------------------
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input_directory', type = str,
		required = True, 
		help = 'path to the output files for processing')
	ap.add_argument('-i2', '--input_directory2', type=str, required=False, default=None,
		help='path to second recording (same animal, different recording)')
	ap.add_argument('-p', '--probe', type = str,
		default='npxl', 
		help = 'generates the correct probe map, this can be "A", "AN", or "npxl"')
	ap.add_argument('-ss', '--spike_sorting', type = str,
		default='kilosort4', 
		help = '"kilosort4" or "vision" was used to do the spike sorting')
	ap.add_argument('-f', '--fps', type = int,
		default=30000,  
		help = 'frames per second for data collection')
	ap.add_argument('-plot', '--plot', action='store_true',
		help = 'save output graphs and initial assessments of the data')
	ap.add_argument('-c', '--class_col', type = str,
		default='group', 
		help = 'column name of group tsv that indicates that classification of neuronal dataset')
	ap.add_argument('-class', '--class', type = str,
		default='good',
		help = "which classifications to include: 'all', 'good', 'mua'")
	ap.add_argument('-b', '--blinding', type=lambda x: x.lower() not in ('false', '0', 'no'), default=True,
		help='Blind half the neurons: True (default) or False')
	ap.add_argument('-parallel', '--parallel', action='store_true',
		help='Parallelize Kent distribution fitting across neurons (default: serial)')
	args = vars(ap.parse_args())

	config = configs()

	#necessary paths
	dataloc = args['input_directory']
	if dataloc.endswith('/'):
		dataloc = dataloc[:-1]
	assert os.path.exists(dataloc), 'Could not find: {}'.format(dataloc)
	config.dataloc = dataloc
	config.parentdir = os.path.dirname(dataloc)

	dataloc2 = args['input_directory2']
	if dataloc2 is not None:
		if dataloc2.endswith('/'):
			dataloc2 = dataloc2[:-1]
		assert os.path.exists(dataloc2), 'Could not find: {}'.format(dataloc2)
		# Results go to common parent, named with the common experimental date
		parent0 = os.path.dirname(dataloc)
		parent1 = os.path.dirname(dataloc2)
		common_parent = os.path.commonpath([parent0, parent1])
		# Extract common date prefix from recording directory names (e.g. '2026-03-05')
		import re
		dir0 = os.path.basename(os.path.normpath(dataloc))
		dir1 = os.path.basename(os.path.normpath(dataloc2))
		# Walk up to find the recording date directories (names like 2026-03-05-0)
		for part in dataloc.split(os.sep):
			if re.match(r'\d{4}-\d{2}-\d{2}-\d+', part):
				dir0 = part
				break
		for part in dataloc2.split(os.sep):
			if re.match(r'\d{4}-\d{2}-\d{2}-\d+', part):
				dir1 = part
				break
		# Find common date prefix (e.g. '2026-03-05' from '2026-03-05-0' and '2026-03-05-1')
		date_match0 = re.match(r'(\d{4}-\d{2}-\d{2})', dir0)
		date_match1 = re.match(r'(\d{4}-\d{2}-\d{2})', dir1)
		if date_match0 and date_match1 and date_match0.group(1) == date_match1.group(1):
			results_name = 'results_{}'.format(date_match0.group(1))
		else:
			results_name = 'results'
		config.savedir = os.path.join(common_parent, results_name)
	else:
		import re
		results_name = 'results'
		# Place results as a sister to the experiment date directory (e.g.
		# .../experiment/results_2026-03-12 alongside .../experiment/2026-03-12-0)
		abs_dataloc = os.path.abspath(dataloc)
		parts = abs_dataloc.split(os.sep)
		common_parent = config.parentdir  # fallback
		for i, part in enumerate(parts):
			date_match = re.match(r'(\d{4}-\d{2}-\d{2})', part)
			if date_match:
				results_name = 'results_{}'.format(date_match.group(1))
				common_parent = os.sep.join(parts[:i]) or os.sep
				break
		config.savedir = os.path.join(common_parent, results_name)
	if not os.path.exists(config.savedir):
		os.mkdir(config.savedir)

	plot = args['plot']
	if plot:
		config.imagedir = os.path.join(config.savedir, 'images')
		if not os.path.exists(config.imagedir):
			os.mkdir(config.imagedir)
	else:
		imagedir = None
	config.spikesorting = args['spike_sorting']
	config.class_col = args['class_col']
	config.group = args['class']
	config.probe = args['probe']
	config.rate = args['fps']

	assert config.spikesorting in ['kilosort4', 'vision'], 'Choose a spike sorting program compatible with this pipeline: "kilsort4" or "vision"'

	# HARDCODDED PARAMETERS

	assert os.path.exists(config.stimdir), 'Stimulus directory was not found.'
	textoutput = os.path.join(config.savedir, 'data_processing_log.txt')
	with open(textoutput, 'w') as f:

		print('Dataset that will be processed: ', config.dataloc, file=f)
		print('\tStimulations files found at: ', config.stimdir, file=f)
		print('Processed dataset will be saved: ', config.savedir, file=f)
		print('Processing the dataset based on was outputs for {} spike sorting.\n'.format(config.spikesorting), file=f)

		data_raw = data_load(dataloc=config.dataloc, spikesorting=config.spikesorting, \
							class_col=config.class_col, group=config.group, rate=config.rate, f=f)

		data_raw2 = None
		if dataloc2 is not None:
			print('\nLoading second recording: ', dataloc2, file=f)
			data_raw2 = data_load(dataloc=dataloc2, spikesorting=config.spikesorting, \
								class_col=config.class_col, group=config.group, rate=config.rate, f=f)

			# Concatenate neurons from both recordings
			nNeu_rec0 = data_raw.asdf.shape[0]
			combined_asdf = np.concatenate([data_raw.asdf, data_raw2.asdf])

			# Offset recording 2 IDs to avoid collisions
			id_offset = int(data_raw.IDs.max()) + 1
			combined_IDs = np.concatenate([data_raw.IDs, data_raw2.IDs + id_offset])

			# Merge cluster DataFrames with same ID offset and dataset label
			if config.spikesorting == 'kilosort4':
				cluster0 = data_raw.cluster.copy()
				cluster0['dataset'] = 0
				cluster2 = data_raw2.cluster.copy()
				cluster2['cluster_id'] = cluster2['cluster_id'] + id_offset
				cluster2['dataset'] = 1
				combined_cluster = pd.concat([cluster0, cluster2], ignore_index=True)

			# Combine probe maps: probeMapFromMeta already returns (span, depth)
			# format matching channel_positions.npy — no column swap needed.
			pm0 = data_raw.chanposition.copy()
			pm1 = data_raw2.chanposition.copy()
			combined_chanposition = np.concatenate([pm0, pm1], axis=0)

			# Offset rec1 channel indices in cluster so they index into combined_chanposition
			ch_offset = pm0.shape[0]
			combined_cluster.loc[combined_cluster['dataset'] == 1, 'ch'] = \
				combined_cluster.loc[combined_cluster['dataset'] == 1, 'ch'] + ch_offset
			if 'most_active_channel' in combined_cluster.columns:
				combined_cluster.loc[combined_cluster['dataset'] == 1, 'most_active_channel'] = \
					combined_cluster.loc[combined_cluster['dataset'] == 1, 'most_active_channel'] + ch_offset

			print('Recording 0: {} neurons, Recording 1: {} neurons, Combined: {}'.format(
				nNeu_rec0, data_raw2.asdf.shape[0], len(combined_IDs)), file=f)
			print('Combined probe map: {} channels, depth range {:.0f}-{:.0f} um'.format(
				combined_chanposition.shape[0], combined_chanposition[:, 1].min(),
				combined_chanposition[:, 1].max()), file=f)
		else:
			nNeu_rec0 = None
			combined_asdf = data_raw.asdf
			combined_IDs = data_raw.IDs
			combined_chanposition = data_raw.chanposition.copy()
			if config.spikesorting == 'kilosort4':
				combined_cluster = data_raw.cluster.copy()
				combined_cluster['dataset'] = 0

		# Save probe map to results folder
		np.save(os.path.join(config.savedir, 'probe_map.npy'), combined_chanposition)
		combined_cluster.to_csv(os.path.join(config.savedir, 'mapped_info.csv'), index=False)
		print('Probe map saved to: {}'.format(os.path.join(config.savedir, 'probe_map.npy')), file=f)
		print('Cluster info saved to: {}'.format(os.path.join(config.savedir, 'mapped_info.csv')), file=f)

		# Save templates to results folder.
		# For dual-recording, build a combined array: shape (id_offset+n1, T, 2*C)
		# so that cluster IDs and channel indices from combined_cluster index correctly.
		if config.spikesorting == 'kilosort4':
			if data_raw2 is not None:
				tmpl0 = data_raw.templates   # (n0, T, C)
				tmpl1 = data_raw2.templates  # (n1, T, C)
				n0, T, C = tmpl0.shape
				n1 = tmpl1.shape[0]
				combined_tmpl = np.zeros((id_offset + n1, T, 2 * C), dtype=tmpl0.dtype)
				combined_tmpl[:n0, :, :C] = tmpl0
				combined_tmpl[id_offset:id_offset + n1, :, C:] = tmpl1
			else:
				combined_tmpl = data_raw.templates
			np.save(os.path.join(config.savedir, 'templates.npy'), combined_tmpl)
			print('Templates saved to: {}'.format(os.path.join(config.savedir, 'templates.npy')), file=f)

		if plot:
			if data_raw2 is not None:
				# Build a combined object for the probe plots
				class _combined_data:
					pass
				combined_data = _combined_data()
				combined_data.spikesorting = config.spikesorting
				combined_data.class_col = config.class_col
				combined_data.chanposition = combined_chanposition
				combined_data.cluster = combined_cluster.copy()
				plot_neurons_relative_to_probe(data_obj=combined_data, save_image_dir=config.imagedir)
				plot_overall_fr_on_probe(data_obj=combined_data, save_image_dir=config.imagedir, good_neuron_ids=combined_IDs)
			else:
				plot_neurons_relative_to_probe(data_obj=data_raw, save_image_dir=config.imagedir)
				plot_overall_fr_on_probe(data_obj=data_raw, save_image_dir=config.imagedir, good_neuron_ids=combined_IDs)

		print('\nConfig parameters used for this data analysis', file=f)
		config.write_attributes(f)
		
		for key, value in config.stim_dict.items():			
			print('New trial:', key, value, file=f)
			file = value[0]	
			fit = value[1]
			multiplier = value[2]
			seg = int(key[-1:])
			print('Working on {} segment, fitting the data to {} distribution. Data stimulation file used is: '.format(seg, fit, file))

			if len(key) > 4:
				seg = int(key[-2:])
			
			if value[1] == 'RandomChord':
				if '4810' in file:
					tn = 48; # tone nums
					tonedur = 10;
					fname = 'RandomChordStim_purerand4810';
				elif '4820' in file:
					tn = 48;
					tonedur = 20;
					fname = 'RandomChordStim_purerand4820';
				elif '4805' in file:
					tn = 48;
					tonedur = 20;
					fname = 'RandomChordStim_purerand4805';
				else:
					tn = 24;
					tonedur = 25;
					fname = 'RandomChordStim_purerand2405';

				tonesdir = os.path.join(config.stimdir, file)
				tonesdirlist = os.listdir(tonesdir)

				print('Number of tones: ', tn, file=f)
				print('\nTone duration: ', tonedur, file=f)
				print('\nNumber of tones files found:, ', len(tonesdirlist), file=f)
				print(fname, file=f)

				pattern_create = False
			else:
				#Read stim file	
				stims, num_stim, stim_ind = readStimFile(config.stimdir, file)

				if len(num_stim) != len(stim_ind):
					n_stim = 1
					for stim in num_stim:
						n_stim *= len(stim)
					print('Number of stimulations: ', n_stim, file=f)
					print(num_stim, file=f)
				else:
					print('Number of stimulations: ', len(num_stim), file=f)
					print(num_stim, file=f)

				print('\nStimulations order: ', stims.shape, file=f)

				print('\nUnique indices: ', stim_ind.shape, file=f)

				pattern_create = True

			#get the ttls of interest
			ttls = getTTLseg(seg=seg, ttls=data_raw.ttl_times, datasets=data_raw.datasep)
			nmult = len(multiplier)
			nttls = len(ttls)
			nNeu = combined_asdf.shape[0]

			# Blinding setup
			blinding_arg = args['blinding']
			blinding_enabled = blinding_arg
			blinded_indices = np.array([], dtype=int)
			processed_indices = np.arange(nNeu)

			if blinding_enabled:
				blinding_file = os.path.join(config.savedir, 'blinding.yml')

				if os.path.exists(blinding_file):
					with open(blinding_file, 'r') as bf:
						blinding_data = yaml.safe_load(bf)
					processed_indices = np.array(blinding_data['processed_indices'])
					blinded_indices = np.array(blinding_data['blinded_indices'])
					print('Loaded existing blinding from: {}'.format(blinding_file), file=f)
				else:
					all_indices = np.arange(nNeu)
					np.random.shuffle(all_indices)
					half = nNeu // 2
					processed_indices = np.sort(all_indices[:half])
					blinded_indices = np.sort(all_indices[half:])
					
					blinding_data = {
						'blinding': True,
						'n_total': int(nNeu),
						'n_processed': int(len(processed_indices)),
						'n_blinded': int(len(blinded_indices)),
						'processed_indices': processed_indices.tolist(),
						'blinded_indices': blinded_indices.tolist(),
						'cluster_ids_processed': combined_IDs[processed_indices].tolist(),
						'cluster_ids_blinded': combined_IDs[blinded_indices].tolist(),
					}
					with open(blinding_file, 'w') as bf:
						yaml.dump(blinding_data, bf, default_flow_style=False)
					print('Blinding file saved to: {}'.format(blinding_file), file=f)

				print('Blinding: {}/{} neurons processed, {}/{} blinded'.format(
					len(processed_indices), nNeu, len(blinded_indices), nNeu), file=f)

			nWin = config.windows.shape[0]

			for m, mult in enumerate(multiplier):
				if nmult == 1:
					print('Using all TTLs to determine pattern, based on 1 multiplier: ' + str(mult), file=f)
					subset_ttls = ttls
				else:
					print('Using {0} TTLs to determine pattern, based on {1}/{2} multipliers: {3}'.format(nttls/nmult, m+1, nmult, mult), file=f)
					ttls_by_mult = np.zeros(nmult)
					if (nttls%nmult) == 0:
						ttlsmult = int(nttls/nmult)
						ttls_by_mult =  ttls_by_mult + ttlsmult
						start = int(m * ttlsmult)
						end = int((m + 1) * ttlsmult)
						subset_ttls = ttls[start:end]
					else:
						if m == 0:
							remainingttl = nttls
							nstims = np.prod(stims.shape)
							print(nstims)
							ttls_by_mult = np.zeros(nmult)
							for j in range(nmult):
								remainingttl -= nstims
								if remainingttl > 0:
									ttls_by_mult[j] = int(nstims)
								else:
									ttls_by_mult[j] = int(nstims + remainingttl)
							print(ttls_by_mult)
							subset_ttls = ttls[:int(ttls_by_mult[m])]
						if (m+1)==nmult:
							subset_ttls = ttls[int(-ttls_by_mult[m]):]
						else:
							start = int(np.sum(ttls_by_mult[:m]))
							end = int(np.sum(ttls_by_mult[:(m+1)]))
							subset_ttls = ttls[start:end]
				
				if pattern_create:
					#get the pattern of this dataset
					if data_raw2 is not None:
						# Generate patterns per-recording using each recording's own TTLs/datasep
						ttls_0 = getTTLseg(seg=seg, ttls=data_raw.ttl_times, datasets=data_raw.datasep)
						if nmult > 1:
							nttls_0 = len(ttls_0)
							if (nttls_0 % nmult) == 0:
								ttlsmult_0 = int(nttls_0 / nmult)
								subset_ttls_0 = ttls_0[int(m * ttlsmult_0):int((m + 1) * ttlsmult_0)]
							else:
								nstims_0 = np.prod(stims.shape)
								remainingttl_0 = nttls_0
								ttls_by_mult_0 = np.zeros(nmult)
								for j in range(nmult):
									remainingttl_0 -= nstims_0
									if remainingttl_0 > 0:
										ttls_by_mult_0[j] = int(nstims_0)
									else:
										ttls_by_mult_0[j] = int(nstims_0 + remainingttl_0)
								if m == 0:
									subset_ttls_0 = ttls_0[:int(ttls_by_mult_0[m])]
								elif (m + 1) == nmult:
									subset_ttls_0 = ttls_0[int(-ttls_by_mult_0[m]):]
								else:
									start_0 = int(np.sum(ttls_by_mult_0[:m]))
									end_0 = int(np.sum(ttls_by_mult_0[:(m + 1)]))
									subset_ttls_0 = ttls_0[start_0:end_0]
						else:
							subset_ttls_0 = ttls_0
						pattern_0, ttlarray = patternGen(data_raw.asdf, subset_ttls_0, stims, num_stim,
														 data_raw.datasep[seg], window=config.timewindow, force=False)

						ttls_1 = getTTLseg(seg=seg, ttls=data_raw2.ttl_times, datasets=data_raw2.datasep)
						if nmult > 1:
							nttls_1 = len(ttls_1)
							if (nttls_1 % nmult) == 0:
								ttlsmult_1 = int(nttls_1 / nmult)
								subset_ttls_1 = ttls_1[int(m * ttlsmult_1):int((m + 1) * ttlsmult_1)]
							else:
								nstims_1 = np.prod(stims.shape)
								remainingttl_1 = nttls_1
								ttls_by_mult_1 = np.zeros(nmult)
								for j in range(nmult):
									remainingttl_1 -= nstims_1
									if remainingttl_1 > 0:
										ttls_by_mult_1[j] = int(nstims_1)
									else:
										ttls_by_mult_1[j] = int(nstims_1 + remainingttl_1)
								if m == 0:
									subset_ttls_1 = ttls_1[:int(ttls_by_mult_1[m])]
								elif (m + 1) == nmult:
									subset_ttls_1 = ttls_1[int(-ttls_by_mult_1[m]):]
								else:
									start_1 = int(np.sum(ttls_by_mult_1[:m]))
									end_1 = int(np.sum(ttls_by_mult_1[:(m + 1)]))
									subset_ttls_1 = ttls_1[start_1:end_1]
						else:
							subset_ttls_1 = ttls_1
						pattern_1, _ = patternGen(data_raw2.asdf, subset_ttls_1, stims, num_stim,
												  data_raw2.datasep[seg], window=config.timewindow, force=False)

						# Concatenate along neuron axis
						pattern = np.concatenate([pattern_0, pattern_1], axis=0)
					else:
						pattern, ttlarray = patternGen(data_raw.asdf, subset_ttls, stims, num_stim,
													   data_raw.datasep[seg],  window=config.timewindow, force=False)

					print('\nNumber of windows assessed: ', config.windows.shape[0], file=f)
					for w, win in enumerate(config.windows): #in ms
						print('\tWindow {0}: {1} - {2}ms'.format(w, win[0],win[1]), file=f)

					#identify which cells are auditory responsive
					activity_df = sigAudFRCompareSpont(pattern=np.squeeze(pattern[:,:,:,0]), spont_win=config.spont_win, 
													   windows=config.windows, test=config.testdist, 
													   siglvl=config.siglvl, minspike=stims.size)

					# activity_df2, IFR, vecTime = sigAudFR_zeta_pvalue(asdf, subset_ttls, 
					# 													datasep, seg, stim_dur=100, 
					# 													boolPlot = False, boolReturnRate=True)
					# activity_df[activity_df2.columns] = activity_df2.values

					activity_df['cluster'] = combined_IDs
					if data_raw2 is not None:
						activity_df['dataset'] = np.concatenate([
							np.zeros(data_raw.asdf.shape[0], dtype=int),
							np.ones(data_raw2.asdf.shape[0], dtype=int)])
					else:
						activity_df['dataset'] = 0

					aud_poisson =  np.array([[None] * nWin] * nNeu)
					aud_qpoisson =  np.array([[None] * nWin] * nNeu)

					for w in range(nWin):
						# aud_poisson[:,w] = (activity_df['poisson win {} neg'.format(str(w))] + 
						# 					activity_df['poisson win {} pos'.format(str(w))]).values
						aud_qpoisson[:,w] = (activity_df['quasipoisson win {} neg'.format(str(w))] +
											 activity_df['quasipoisson win {} pos'.format(str(w))]).values

					if blinding_enabled:
						for idx in blinded_indices:
							aud_qpoisson[idx, :] = 0
							aud_poisson[idx, :] = 0

					seqsavedir = os.path.join(config.savedir, '{0} mult {1}/'.format(key, mult))
					if not os.path.exists(seqsavedir):
						os.mkdir(seqsavedir)
						if plot:
							pattern_plot_dir = os.path.join(seqsavedir, 'images')
							try:
								os.mkdir(pattern_plot_dir)
							except:
								print('')	
					np.save(os.path.join(seqsavedir, 'pattern.npy'), pattern)
					np.save(os.path.join(seqsavedir, 'good_neurons.npy'), combined_IDs)
					np.save(os.path.join(seqsavedir, 'aud_neurons_poisson.npy'), aud_poisson) 
					np.save(os.path.join(seqsavedir, 'windows.npy'), config.windows)
					np.save(os.path.join(seqsavedir, 'spont_win.npy'), config.spont_win)
					np.save(os.path.join(seqsavedir, 'aud_neurons_qpoisson.npy'), aud_qpoisson)  
					# np.save(os.path.join(seqsavedir, 'InstantaneousFR.npy'), IFR) #if we want to save the instantaeous FR
					# np.save(os.path.join(seqsavedir, 'InstantaneousVecTime.npy'), vecTime)
					activity_df.to_csv(os.path.join(seqsavedir, 'activity-df.csv'))
					
					print('Preprocessing data saved to :', seqsavedir, file=f)

					if plot:
						plot_img_dir = os.path.join(seqsavedir, 'images')
						if not os.path.exists(plot_img_dir):
							os.mkdir(plot_img_dir)
						if data_raw2 is not None:
							plot_windowed_data = combined_data
						else:
							plot_windowed_data = data_raw
						_cm = plot_windowed_data.cluster.copy()
						for w_plot, win_plot in enumerate(config.windows):
							ave_col = "avg window {0} - {1} ms".format(win_plot[0], win_plot[1])
							if ave_col in activity_df.columns:
								_cm[f"windowed_fr_win{w_plot}"] = float("nan")
								for _idx, _cid in enumerate(combined_IDs):
									_cm.loc[_cm["cluster_id"] == _cid, f"windowed_fr_win{w_plot}"] = activity_df[ave_col].iloc[_idx]
							_cm.attrs[f"window_{w_plot}"] = list(win_plot)
							plot_windowed_fr_on_probe(_cm, plot_windowed_data.chanposition,
													  save_image_dir=plot_img_dir,
													  window_index=w_plot)

					#Fitting the data, only process select neurons
					window_summaries = {}  # collect per-window RF summary for cluster_map.csv
					for w in range(nWin):

						print('Auditory neurons are determined based by quasipoisson statistics.')
						neuron_assess = np.where(aud_qpoisson[:,w]==1)[0]
						if blinding_enabled:
							neuron_assess = np.intersect1d(neuron_assess, processed_indices)
						frac2 = len(neuron_assess)/nNeu
						print('\tFraction of auditory responsive neurons {0}/{1}: {2}%'.format(len(neuron_assess), nNeu,
																  np.round(frac2*100, 2)), file=f)

						# Apply minimum total AP count criterion: neuron must have > 25 spikes
						# summed across ALL positions and trials in the analysis window.
						print('\tCalculating counts for the window size', file=f)
						data, _ = PatternToCount(pattern=pattern, timerange=list(config.windows[w]),
												 timeBinSz=np.diff(config.windows[w])[0])
						total_aps = data.sum(axis=tuple(range(1, data.ndim)))
						min_ap = getattr(config, 'min_ap_count', 25)
						ap_mask = total_aps > min_ap
						n_before = len(neuron_assess)
						neuron_assess = np.intersect1d(neuron_assess, np.where(ap_mask)[0])
						print('\tAP threshold (>{} APs): {}/{} auditory neurons pass'.format(
							min_ap, len(neuron_assess), n_before), file=f)
						print('\tAP threshold (>{} APs): {}/{} auditory neurons pass'.format(
							min_ap, len(neuron_assess), n_before))

						if fit == 'kent':
							param_labels = ['kappa', 'beta', 'theta', 'phi', 'alpha', 'height']

							elev = 90-np.array(num_stim[0]).astype(int)
							azim = -1*np.array(num_stim[1]).astype(int)
							azim = azim*np.pi/180
							elev = elev*np.pi/180
							try:
								laser = np.array(num_stim[2]).astype(int)
								parameters = np.zeros([nNeu, 2, len(param_labels)])
								variances = np.zeros([nNeu, 2, len(param_labels)])
								modelfits = np.zeros([nNeu, 2, len(elev), len(azim)])
								datafits = modelfits.copy()
								aic_bic = np.zeros((nNeu,2,4))
								sumresid = np.zeros((nNeu,2,2))
							except:
								laser = None
								parameters = np.zeros([nNeu, len(param_labels)])
								variances = np.zeros([nNeu, len(param_labels)])
								modelfits = np.zeros([nNeu, len(elev), len(azim)])
								datafits = modelfits.copy()
								aic_bic = np.zeros((nNeu,4))
								sumresid = np.zeros((nNeu,2))

							print('\nFitting Kent distribution', file=f)

							# Build argument list for fitting
							fit_args = [(n, data[n], azim, elev, laser) for n in neuron_assess]
							n_total = len(neuron_assess)
							print_interval = max(1, n_total // 10)

							use_parallel = args.get('parallel', False)
							if use_parallel:
								n_workers = min(cpu_count() - 1 or 1, n_total)
								print('Fitting {} neurons in parallel with {} workers'.format(
									n_total, n_workers), file=f)
								results_list = []
								with Pool(n_workers) as pool:
									for i, result in enumerate(pool.imap_unordered(_fit_neuron_kent, fit_args), 1):
										results_list.append(result)
										if i % print_interval == 0 or i == n_total:
											print('  Fitting progress: {}/{} neurons ({:.0f}%)'.format(
												i, n_total, 100*i/n_total), flush=True)
							else:
								print('Fitting {} neurons serially'.format(n_total), file=f)
								results_list = []
								for i, args_tuple in enumerate(fit_args, 1):
									results_list.append(_fit_neuron_kent(args_tuple))
									if i % print_interval == 0 or i == n_total:
										print('  Fitting progress: {}/{} neurons ({:.0f}%)'.format(
											i, n_total, 100*i/n_total), flush=True)

							# Unpack results into output arrays
							for n, res in results_list:
								if laser is not None:
									for l in laser:
										parameters[n, l] = res[l]['params']
										variances[n, l] = res[l]['var']
										modelfits[n, l] = res[l]['fitdist']
										datafits[n, l] = res[l]['data']
										aic_bic[n, l] = res[l]['aic_bic']
										sumresid[n, l] = res[l]['sumresid']
								else:
									parameters[n] = res['params']
									variances[n] = res['var']
									modelfits[n] = res['fitdist']
									datafits[n] = res['data']
									aic_bic[n] = res['aic_bic']
									sumresid[n] = res['sumresid']

							print('Fitting complete for {} neurons'.format(len(neuron_assess)), file=f)
							winsavedir = os.path.join(seqsavedir, '{0} fit win {1}/'.format(fit, w))
							if not os.path.exists(winsavedir):
								os.mkdir(winsavedir)
							np.save(os.path.join(winsavedir,'neuron_assess.npy'), neuron_assess)
							np.save(os.path.join(winsavedir, 'parameters.npy'), parameters)
							np.save(os.path.join(winsavedir, 'variances.npy'), variances)   
							np.save(os.path.join(winsavedir, 'modelfits.npy'), modelfits) 
							np.save(os.path.join(winsavedir, 'datafits.npy'), datafits)     
							np.save(os.path.join(winsavedir, 'aic_bic.npy'), aic_bic)
							np.save(os.path.join(winsavedir, 'sumresid.npy'), sumresid)
							
							print('Kent distribution data saved to :', winsavedir, file=f)

							# RF neuron identification: Kent BIC < uniform BIC
							# aic_bic columns: [uniform_aic, kent_aic, uniform_bic, kent_bic]
							bic_uniform = aic_bic[:, 2] if aic_bic.ndim == 2 else aic_bic[:, 0, 2]
							bic_kent = aic_bic[:, 3] if aic_bic.ndim == 2 else aic_bic[:, 0, 3]
							delta_bic = bic_uniform - bic_kent
							is_rf = delta_bic > 0
							rf_indices = np.where(is_rf)[0]
							np.save(os.path.join(winsavedir, 'is_rf.npy'), is_rf)
							np.save(os.path.join(winsavedir, 'rf_indices.npy'), rf_indices)
							
							if blinding_enabled:
								print(f'  Window {w}: {is_rf.sum()} / {blinding_data["n_processed"]} RF blinded neurons', file=f)
							else:
								print(f'  Window {w}: {is_rf.sum()} / {len(is_rf)} RF neurons', file=f)
							win_params = parameters if parameters.ndim == 2 else parameters[:, 0, :]
							window_summaries[w] = {
								'is_rf': is_rf,
								'delta_bic': delta_bic,
								'parameters': win_params,
								'neuron_assess': neuron_assess,
							}

					# Build and save cluster_map.csv with RF designations and preferred positions
					if fit == 'kent' and window_summaries:
						cluster_map_df = pd.DataFrame({'cluster_id': combined_IDs})
						if data_raw2 is not None:
							cluster_map_df['dataset'] = np.concatenate([
								np.zeros(data_raw.asdf.shape[0], dtype=int),
								np.ones(data_raw2.asdf.shape[0], dtype=int)])
						else:
							cluster_map_df['dataset'] = 0
						for w in sorted(window_summaries.keys()):
							ws = window_summaries[w]
							cluster_map_df[f'is_auditory_win{w}'] = aud_qpoisson[:, w].astype(bool)
							cluster_map_df[f'is_rf_win{w}'] = ws['is_rf']
							pref_az = np.degrees(ws['parameters'][:, 3]).copy()
							pref_el = (90 - np.degrees(ws['parameters'][:, 2])).copy()
							fitted_mask = np.zeros(nNeu, dtype=bool)
							fitted_mask[ws['neuron_assess']] = True
							pref_az[~fitted_mask] = np.nan
							pref_el[~fitted_mask] = np.nan
							cluster_map_df[f'pref_azimuth_deg_win{w}'] = pref_az
							cluster_map_df[f'pref_elevation_deg_win{w}'] = pref_el
							cluster_map_df[f'delta_bic_win{w}'] = ws['delta_bic']
						cluster_map_df.to_csv(os.path.join(seqsavedir, 'cluster_map.csv'), index=False)
						print('Cluster map saved to :', os.path.join(seqsavedir, 'cluster_map.csv'), file=f)

						# Save a config file with the info needed to run plot_rf_neurons.py
						abs_seqsavedir = os.path.abspath(seqsavedir.rstrip('/'))
						abs_kilosort_loc = os.path.abspath(config.dataloc)
						plot_rf_config = {
							'input_dir': abs_seqsavedir,
							'spikesorting': config.spikesorting,
							'kilosort_loc': abs_kilosort_loc,
							'window': 0,
							'selection': 'rf',
							'command': 'python plot_rf_neurons.py -i "{}" -ss {} -k "{}"'.format(
								abs_seqsavedir, config.spikesorting, abs_kilosort_loc),
						}
						plot_rf_config_path = os.path.join(seqsavedir, 'plot_rf_neurons_config.json')
						with open(plot_rf_config_path, 'w') as jf:
							json.dump(plot_rf_config, jf, indent=2)
						print('RF plot config saved to :', plot_rf_config_path, file=f)

				elif fit == 'RandomChord':
					seqsavedir = os.path.join(config.savedir, '{0} mult {1}/'.format(key, mult))
					if not os.path.exists(seqsavedir):
						os.mkdir(seqsavedir)

					# Load picked_tones from stimulus .mat file
					stim_mat_path = os.path.join(config.stimdir, fname + '.mat')
					pt = sio.loadmat(stim_mat_path)
					picked_tones = pt['picked_tones']

					nreps = mult
					print('Running RandomChord STA analysis (nreps={}, tn={})'.format(nreps, tn), file=f)
					if data_raw2 is not None:
						# Run RandomChord per-recording with each recording's own TTLs/datasep
						ttls_0 = getTTLseg(seg=seg, ttls=data_raw.ttl_times, datasets=data_raw.datasep)
						results_0 = DoRandomChordAnalysis(
							data_raw.asdf, ttls_0, nreps,
							picked_tones, tn, data_raw.datasep[seg], data_raw.asdf.shape[0]
						)
						ttls_1 = getTTLseg(seg=seg, ttls=data_raw2.ttl_times, datasets=data_raw2.datasep)
						results_1 = DoRandomChordAnalysis(
							data_raw2.asdf, ttls_1, nreps,
							picked_tones, tn, data_raw2.datasep[seg], data_raw2.asdf.shape[0]
						)
						# Concatenate results
						results = {
							'stas': list(results_0['stas']) + list(results_1['stas']),
							'stasigs': list(results_0['stasigs']) + list(results_1['stasigs']),
							'nSpikes': np.concatenate([results_0['nSpikes'], results_1['nSpikes']]),
							'meanpic': (results_0['meanpic'] + results_1['meanpic']) / 2,
						}
					else:
						results = DoRandomChordAnalysis(
							data_raw.asdf, subset_ttls, nreps,
							picked_tones, tn, data_raw.datasep[seg], nNeu
						)

					if blinding_enabled:
						for idx in blinded_indices:
							results['stas'][idx] = np.zeros_like(results['stas'][idx]) if results['stas'][idx] is not None else None
							results['stasigs'][idx] = np.zeros_like(results['stasigs'][idx]) if results['stasigs'][idx] is not None else None
							results['nSpikes'][idx] = 0

					np.save(os.path.join(seqsavedir, 'stas.npy'), np.array(results['stas'], dtype=object))
					np.save(os.path.join(seqsavedir, 'stasigs.npy'), np.array(results['stasigs'], dtype=object))
					np.save(os.path.join(seqsavedir, 'nSpikes.npy'), results['nSpikes'])
					np.save(os.path.join(seqsavedir, 'meanpic.npy'), results['meanpic'])

					print('RandomChord analysis data saved to:', seqsavedir, file=f)
