#!/usr/bin/env python3
'''
Full pipeline for analysis. All custom functions used are found in ./python

Authors: Brian R Mullen
Date: 2026-02-10
'''


import os
import math

from scipy.stats import poisson
from scipy.stats import nbinom
import scipy.io as sio

# custom functions
from python.preprocessing import probeMap, readStimFile, getTTLseg, patternGen, PatternToCount, sigAudFRCompareSpont
from python.distributions_fit import azimElevCoord, kent_fit, uniform_fit, aic_leastsquare, bic_leastsquare
from python.visualizations import plot_neurons_relative_to_probe
from python.random_chord_analysis import DoRandomChordAnalysis
from config import configs
from python.hdf5manager import hdf5manager as h5

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
			self.cluster = cluster[['cluster_id', self.class_col, 'ch', 'depth', 'fr']]
			
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
	ap.add_argument('-idict', '--input_dict', type = str,
		required = False, #THIS WILL ULTIMATELY BE TRUE
		help = 'path to stimulus dictionary files, indicating what the data is and how it will be processed')
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
		help = 'if flagged, it will save output graphs and initial assessments of the data')
	ap.add_argument('-c', '--class_col', type = str,
		default='group', 
		help = 'column name of group tsv that indicates that classification of neuronal dataset')
	ap.add_argument('-class', '--class', type = str,
		default='good', 
		help = "which classifications to include: 'all', 'good', 'mua'")
	args = vars(ap.parse_args())

	config = configs()

	#necessary paths
	dataloc = args['input_directory']
	if dataloc.endswith('/'):
		dataloc = dataloc[:-1]
	assert os.path.exists(dataloc), 'Could not find: {}'.format(dataloc)
	config.dataloc = dataloc
	config.parentdir = os.path.dirname(dataloc)
	config.savedir = os.path.join(config.parentdir, 'results')
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
		
		if plot:
			plot_neurons_relative_to_probe(data_obj=data_raw, save_image_dir=config.imagedir)

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
			nNeu = data_raw.asdf.shape[0]
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

					activity_df['cluster'] = data_raw.IDs

					aud_poisson =  np.array([[None] * nWin] * nNeu)
					aud_qpoisson =  np.array([[None] * nWin] * nNeu)

					for w in range(nWin):
						# aud_poisson[:,w] = (activity_df['poisson win {} neg'.format(str(w))] + 
						# 					activity_df['poisson win {} pos'.format(str(w))]).values
						aud_qpoisson[:,w] = (activity_df['quasipoisson win {} neg'.format(str(w))] + 
											 activity_df['quasipoisson win {} pos'.format(str(w))]).values
					
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
					np.save(os.path.join(seqsavedir, 'good_neurons.npy'), data_raw.IDs)   
					np.save(os.path.join(seqsavedir, 'aud_neurons_poisson.npy'), aud_poisson) 
					np.save(os.path.join(seqsavedir, 'windows.npy'), config.windows)
					np.save(os.path.join(seqsavedir, 'spont_win.npy'), config.spont_win)
					np.save(os.path.join(seqsavedir, 'aud_neurons_qpoisson.npy'), aud_qpoisson)  
					# np.save(os.path.join(seqsavedir, 'InstantaneousFR.npy'), IFR) #if we want to save the instantaeous FR
					# np.save(os.path.join(seqsavedir, 'InstantaneousVecTime.npy'), vecTime)
					activity_df.to_csv(os.path.join(seqsavedir, 'activity-df.csv'))
					
					print('Preprocessing data saved to :', seqsavedir, file=f)

					#Fitting the data, only process select neurons
					for w in range(nWin):

						print('Auditory neurons are determined based by quasipoisson statistics.')
						neuron_assess = np.where(aud_qpoisson[:,w]==1)[0]
						frac2 = len(neuron_assess)/nNeu
						print('\tFraction of auditory responsive neurons {0}/{1}: {2}%'.format(len(neuron_assess), nNeu, 
																  np.round(frac2*100, 2)), file=f)
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

							print('\tCalculating counts for the window size', file=f)
							data, _ = PatternToCount(pattern=pattern,timerange=list(config.windows[w]), 
												  timeBinSz=np.diff(config.windows[w])[0])

							for n in neuron_assess:
								if laser is not None:            
									print('\nWorking on Optogenetics neuron {}'.format(n))
									for l in laser:
										if l == 0:
											mean_data = np.mean(np.squeeze(data[n,:,:,0]), axis=2)
										else: 
											mean_data = np.mean(np.squeeze(data[n,:,:,1]), axis=2)

										xs = azimElevCoord(azim, elev, np.mean(np.squeeze(data[n]), axis=2))
										xyz = xs[:,:3]
										aud = xs[:,3]
										kentfit = kent_fit(aud, xyz, datashape=mean_data.shape)
										uniformfit = uniform_fit(aud, np.arange(mean_data.size))
										
										uaic = aic_leastsquare(uniformfit.residuals, uniformfit.params)
										ubic = bic_leastsquare(uniformfit.residuals, uniformfit.params)
										
										kaic = aic_leastsquare(kentfit.residuals, kentfit.params)
										kbic = bic_leastsquare(kentfit.residuals, kentfit.params)

										parameters[n, l]  = kentfit.params
										variances[n, l] = kentfit.var
										modelfits[n, l] = kentfit.fitdist
										datafits[n, l] = kentfit.data
										aic_bic[n, l] = [uaic, kaic, ubic, kbic]
										sumresid[n, l] = [uniformfit.residuals_sum, kentfit.residual_sum]
								else:
									print('\nWorking on neuron {}'.format(n))
									
									mean_data = np.mean(np.squeeze(data[n]), axis=2)

									xs = azimElevCoord(azim, elev, np.mean(np.squeeze(data[n]), axis=2))
									xyz = xs[:,:3]
									aud = xs[:,3]
									kentfit = kent_fit(aud, xyz, datashape=mean_data.shape)
									uniformfit = uniform_fit(aud, np.arange(mean_data.size))
									
									uaic = aic_leastsquare(uniformfit.residuals, uniformfit.params)
									ubic = bic_leastsquare(uniformfit.residuals, uniformfit.params)
									
									kaic = aic_leastsquare(kentfit.residuals, kentfit.params)
									kbic = bic_leastsquare(kentfit.residuals, kentfit.params)

									parameters[n] = kentfit.params
									variances[n] = kentfit.var
									modelfits[n] = kentfit.fitdist
									datafits[n] = kentfit.data
									aic_bic[n] = [uaic, kaic, ubic, kbic]
									sumresid[n] = [uniformfit.residuals_sum, kentfit.residual_sum]

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
							print(f'  Window {w}: {is_rf.sum()} / {len(is_rf)} RF neurons', file=f)

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
					results = DoRandomChordAnalysis(
						data_raw.asdf, subset_ttls, nreps,
						picked_tones, tn, data_raw.datasep[seg], nNeu
					)

					np.save(os.path.join(seqsavedir, 'stas.npy'), np.array(results['stas'], dtype=object))
					np.save(os.path.join(seqsavedir, 'stasigs.npy'), np.array(results['stasigs'], dtype=object))
					np.save(os.path.join(seqsavedir, 'nSpikes.npy'), results['nSpikes'])
					np.save(os.path.join(seqsavedir, 'meanpic.npy'), results['meanpic'])

					print('RandomChord analysis data saved to:', seqsavedir, file=f)
