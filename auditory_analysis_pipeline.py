import os
import sys
import math

sys.path.append('./python')
# sys.path.append('/home/feldheimlab/Documents/pyWholeBrain/')

from scipy.stats import poisson
from scipy.stats import norm
import scipy.io as sio

from preprocessing import *
from distributions_fit import *
from visualizations import *
# from hdf5manager import hdf5manager as h5

from scipy.stats import poisson
from scipy.stats import nbinom

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable



def get_IDs(cluster: pd.DataFrame, 
			class_col: str, 
			group: str = 'good'):
	'''
	Gets the cluster of IDs of the classification indicated
	
	Arguments:
		cluster: pd.DataFrame the indicates classification of clusters (usually saved from phy)
		class_col: column in the pd.DataFrame that the classifcation is in
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
	def __init__(self, dataloc, spikesorting, class_col, group):
		
		self.spikesorting = spikesorting
		parentdir = os.path.dirname(dataloc)
		self.class_col = class_col
		self.group = group

		if self.spikesorting  == 'vision':
			#VISION SPECIFIC
			asdfpath = os.path.join(dataloc, 'asdf_orig.mat')
			risedpath = os.path.join(dataloc, 'ttlTimes.mat')
			basicinfopath = os.path.join(dataloc, 'xy.mat')
			segmentlegths = os.path.join(dataloc, 'segmentlengths.mat')
			segmentstims = os.path.join(dataloc, 'segmentstims.mat')
			segment = os.path.join(dataloc, 'segttls.mat')

			self.basicinfo = sio.loadmat(basicinfopath)#.load()
			self.ttl_times = np.squeeze(sio.loadmat(risedpath)['ttlTimes'])
			segttls = sio.loadmat(segment)#.load()
			# segmentstim = sio.loadmat(segmentstims)#.load()

			datasep = np.squeeze(sio.loadmat(segmentlegths)['segmentseparations'])
			datasep2 = np.zeros(datasep.shape[0]+1)
			datasep2[1:] = datasep
			self.datasep = datasep2

			asdf_raw = sio.loadmat('asdf.mat')['asdf_raw']
			# aud_ids = sio.loadmat(os.path.join(dataloc, 'AuditorySpotSummary_1-posneu.mat'))['posneu']

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
				self.fr[n] = len(asdf[n])/datasep[-1]*1000


		elif self.spikesorting  == 'kilosort4':
			#spikes
			spiketemplates = np.load(os.path.join(dataloc, 'spike_clusters.npy')) # to make asdf
			spiketimes = np.load(os.path.join(dataloc, 'spike_times.npy')) # to make asdf
			templates = np.load(os.path.join(dataloc, 'templates.npy')) #waveforms
			try:
				cluster = pd.read_csv(os.path.join(dataloc, 'cluster_info.tsv'), sep='\t') #class 
			except:
				cluster = pd.read_csv(os.path.join(dataloc, 'cluster_group.tsv'), sep='\t')
			
			cluster.dropna(subset=[class_col], inplace=True)
			self.cluster = cluster[['cluster_id', class_col, 'ch', 'depth', 'fr']]
			
			groups = np.unique(cluster[class_col])

			for group in groups:
				frac = len(cluster[cluster[class_col]==group])/len(cluster)
				print('{0} fraction {1}/{2}: {3}%'.format(group, len(cluster[cluster[class_col]==group]), 
														 len(cluster), np.round(frac*100, 2)), file=f)
			
			if group != 'all':
				cluster = cluster[cluster[class_col]==group]

			#created from rhd files Save the ttl times in a file so we don't have to determine this every time
			# digital = np.load(os.path.join(kilosortloc, 'digital.npy')) #get ttls
			try:
				datasep = np.load(os.path.join(dataloc, 'datasep.npy'), allow_pickle=True).item() #get data separations 
			except:
				datasep = np.load(os.path.join(parentdir, 'datasep.npy'), allow_pickle=True).item()
			
			self.datasep = datasep['Datasep']
			
			self.IDs, IDs_index = get_IDs(cluster, class_col=class_col, group=group)

			print('\nProcessing only "good" neurons: ', file=f)
			print(cluster.head(), file=f)

			#load channel maps
			self.chanmap = np.load(os.path.join(dataloc,'channel_map.npy')) #map of the probe
			self.chanposition = np.load(os.path.join(dataloc,'channel_positions.npy'))
			try:
				self.ttl_times = np.load(os.path.join(dataloc, 'ttlTimes.npy'))
			except: 
				self.ttl_times= np.load(os.path.join(parentdir, 'ttlTimes.npy'))

			neuron_clust = np.unique(spiketemplates)
			max_clust = len(cluster)
			self.asdf = np.empty((max_clust), dtype=object)

			for n, clust in enumerate(self.IDs):
				try:
					self.asdf[n]=list(np.squeeze(spiketimes[spiketemplates==clust]))
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
		help = 'generates the correct probemap, this can be "A", "AN", or "npxl"')
	ap.add_argument('-ss', '--spike_sorting', type = str,
		default='kilosort4', 
		help = '"kilosort4" or "vision" was used to do the spike sorting')
	ap.add_argument('-f', '--fps', type = int,
		default=30000,  
		help = 'frames per second for data collection')
	ap.add_argument('-plot', '--plot', action='store_true',
		help = 'if flagged, it will save output graphs and initial assessements of the data')
	ap.add_argument('-c', '--class_col', type = str,
		default='group', 
		help = 'column name of group tsv that indicates that classification of neuronal dataset')
	ap.add_argument('-class', '--class', type = str,
		default='good', 
		help = "which classifications to include: 'all', 'good', 'mua'")
	args = vars(ap.parse_args())


	#necessary paths
	dataloc = args['input_directory']
	if dataloc.endswith('/'):
		dataloc = dataloc[:-1]
	assert os.path.exists(dataloc), 'Could not find: {}'.format(kilosortloc)
	parentdir = os.path.dirname(dataloc)
	savedir = os.path.join(parentdir, 'results')
	if not os.path.exists(savedir):
		os.mkdir(savedir)


	plot = args['plot']
	save = True
	if plot:
		image_dir = os.path.join(savedir, 'images')
		if not os.path.exists(image_dir):
			os.mkdir(image_dir)
	else:
		image_dir = None
	spikesorting = args['spike_sorting']
	class_col = args['class_col']
	group = args['class']
	probe = args['probe']
	rate = args['fps']
	class_col = args['class_col']
	classification = args['class']

	assert spikesorting in ['kilosort4', 'vision'], 'Choose a spike sorting program compatable with this pipeline: "kilsort4" or "vision"'
	
	#MAKE THIS AN INPUT FILE FOR THE PIPELINE!
	stim_dict = {'stimdir':'/Users/ackmanadmin/Documents/test_dataset_auditory_pipeline/python/Auditory/stimgen/npx_gen1/',
				'seg1':['fullfield_newspeakers/fullfield_newspeakers.txt', 'kent', [1, 10]],
				'seg2':['RandomChord/randomchord4810_newspeakers', 'RandomChord', [1]]}

	# HARDCODDED PARAMETERS
	timewindow = [0,1000] #pattern generation window

	#determine auditory responsive neurons
	spont_win = [700,1000]
	windows = np.array([[5,20]])#,[20,100],[105,120]]) #windows
	testdist = ['quasipoisson']#['poisson','quasipoisson', 'nbinom'] #tests
	siglvl = [0.05]#[0.001,0.001,0.001]#[0.001,0.01,0.05] #alpha significance
	nWin = windows.shape[0] #n of windows

	stimdir = stim_dict['stimdir']
	assert os.path.exists(stimdir), 'Stimulus directory was not found.'

	with open(os.path.join(savedir, 'data_processing_log.txt'), 'w') as f:

		print('Dataset that will be processed: ', dataloc, file=f)
		print('\tStimulations files found at: ', stimdir, file=f)
		print('Processed dataset will be saved: ', savedir, file=f)
		print('Processing the dataset based on was outputs for {} spikesorting.'.format(spikesorting), file=f)
		data_raw = data_load(dataloc, spikesorting, class_col, group)
		
		if plot:
			plot_neurons_relative_to_probe(data_obj=data_raw, save_image_dir=image_dir)

		for key, value in stim_dict.items():
			#preprocessing
			if key == 'stimdir':
				continue
			
			print('New trial:', key, value, file=f)
			file = value[0]
			fit = value[1]
			multiplier = value[2]
			seg = int(key[-1:])
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

				tonesdir = os.path.join(stimdir, file)
				tonesdirlist = os.listdir(tonesdir)

				print('Number of tones: ', tn, file=f)
				print('\nTone duration: ', tonedur, file=f)
				print('\nNumber of tones files found:, ', len(tonesdirlist), file=f)
				print(fname, file=f)

				pattern_create = False

			else:
				#Read stim file	
				stims, num_stim, stim_ind = readStimFile(stimdir, file)

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

			for m, mult in enumerate(multiplier):
				if nmult == 1:
					print('Using all TTLs to determine pattern, based on 1 mutlipier: ' + str(mult), file=f)
					subset_ttls = ttls
				else:
					print('Using {0} TTLs to determine pattern, based on {1}/{2} mutlipiers: {3}'.format(nttls/nmult, m+1, nmult, mult), file=f)
					ttlsmult = int(nttls/nmult)
					if m == 0:
						subset_ttls = ttls[:ttlsmult]
					if (m+1)==nmult:
						subset_ttls = ttls[-ttlsmult:]
					else:
						subset_ttls = ttls[m*ttlsmult:(m+1)*ttlsmult]
				
				if pattern_create:

					#get the pattern of this dataset
					pattern, ttlarray = patternGen(data_raw.asdf, subset_ttls, stims, num_stim, 
												   data_raw.datasep[seg],  window=timewindow, force=False)


					print('\nNumber of windows assessed: ', windows.shape[0], file=f)
					for w, win in enumerate(windows): #in ms
						print('\tWindow {0}: {1} - {2}ms'.format(w, win[0],win[1]), file=f)

					#identify which cells are auditory responsive
					activity_df = sigAudFRCompareSpont(pattern=np.squeeze(pattern[:,:,:,0]), spont_win=spont_win, 
													   windows=windows, test=testdist, 
													   siglvl=siglvl, minspike=stims.size)

					'''
					Need to work on getting the sigAudFR_zeta_pvalue.  
					The function worked on the first two neurons (0, 1) but stalled on/did not progress from neuron 2 from 2024-10-20-0
					
					I put boolReturnRate=False to increase the speed of the code.  If we want to save the instantaneous firing rates,
					we should turn it to True
					'''
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
					
					if save:
						seqsavedir = os.path.join(savedir, '{0} mult {1}/'.format(key, mult))
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
						np.save(os.path.join(seqsavedir, 'aud_neurons_qpoisson.npy'), aud_qpoisson)     
						np.save(os.path.join(seqsavedir, 'spont_win'), spont_win)
						np.save(os.path.join(seqsavedir, 'windows'), windows)
						# np.save(os.path.join(seqsavedir, 'InstantaneousFR.npy'), IFR) #if we want to save the instantaeous FR
						# np.save(os.path.join(seqsavedir, 'InstantaneousVecTime.npy'), vecTime)
						activity_df.to_csv(os.path.join(seqsavedir, 'activity-df.csv'))
						
						print('Preprocessing data saved to :', seqsavedir, file=f)

					#Fitting the data, only process select neurons
					for w in range(nWin):

						print('Auditory neurons are deterimed based on quasipoisson statistics.')
						neuron_assess = np.where(aud_qpoisson[:,w]==1)[0] # I USED THE QUASIPOISSON TO DETERMINE WHICH NEURONS TO FIT TO THE DISTRIBUTION
						
						groups = ['good', 'mua', 'noise']
						for group in groups:
							frac = len(data_raw.cluster[data_raw.cluster[class_col]==group])/nNeu
							n_group = len(data_raw.cluster[data_raw.cluster[class_col]==group])
							print('{0} fraction {1}/{2}: {3}%'.format(group, n_group, 
																	  nNeu, np.round(frac*100, 2)), file=f)
							if group == 'good':
								frac2 = len(neuron_assess)/n_group
								print('\tgood and auditory responsive fraction {0}/{1}: {2}%'.format(len(neuron_assess), n_group, 
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

							print('\nFitting kent distribution', file=f)

							print('\tCalculating counts for the window size', file=f)
							data = PatternToCount(pattern=pattern,timerange=list(windows[w]), 
												  timeBinSz=np.diff(windows[w]))

							for n in neuron_assess:
								if laser is not None:            
									print('\nWorking on Optogenetics neuron {}'.format(n))
									for l in laser:
										if l == 0:
											mean_data = np.mean(np.squeeze(data[n,:,:,0]), axis=2)
										else: 
											mean_data = np.mean(np.squeeze(data[n,:,:,1]), axis=2)

										xs = azimElevCoord(azim, elev, np.squeeze(mean_data))
										xyz = xs[:,:3]

										niter = 50 #number of random starts for determining 
										for i in np.arange(niter):
											p, var, residual = fitKent(xs[:,3], xyz)
											fitdist = kentdist(*p, xyz)
											if i == 0:
												param_store = np.zeros((niter, len(p)))
												resid_store = np.zeros(niter)
												var_store = np.zeros((niter, len(p)))
												fun_store = np.zeros(niter)
											resid_store[i] = np.sum(np.abs(residual))
											param_store[i] = p
											var_store[i] = var    
											fun_store[i] = aic_leastsquare(residual, p)

										index = np.nanargmin(fun_store)
										p = param_store[index]        
										var = var_store[index]

										up = fitUniform(mean_data.reshape(mean_data.size), np.arange(mean_data.size))
										ufit = uniform(*up, np.arange(mean_data.size))
										uresidual = ufit.reshape(mean_data.shape)-mean_data

										uaic = aic_leastsquare(uresidual, up)
										ubic = bic_leastsquare(uresidual, up)

										fitdist = kentdist(*p, xyz).reshape(mean_data.shape)
										fresidual = fitdist.reshape(mean_data.shape)-mean_data

										kaic = aic_leastsquare(fresidual, p)
										kbic = bic_leastsquare(fresidual, p)

										kentres = chiSquaredTest(mean_data.reshape(mean_data.size), fitdist.reshape(fitdist.size), p)

										parameters[n, l] = p
										variances[n, l] = var
										modelfits[n, l] = fitdist
										datafits[n, l] = mean_data
										aic_bic[n, l] = [uaic, kaic, ubic, kbic]
										sumresid[n, l] = [np.sum(np.abs(uresidual)), resid_store[index]]

								else:
									print('\nWorking on neuron {}'.format(n))
									
									mean_data = np.mean(np.squeeze(data[n]), axis=2)

									xs = azimElevCoord(azim, elev, np.mean(np.squeeze(data[n]), axis=2))
									xyz = xs[:,:3]
									kentfit = kent_fit(xs[:,3], xyz, datashape=mean_data.shape)
									uniformfit = uniform_fit(xs[:,3], np.arange(mean_data.size))
									
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

							if save:
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
				
				elif fit == 'RandomChord':
					#code here
					print('NEED TO COMPLETE RandomChord', file=f)    
# 					activity_df2, IFR, vecTime = sigAudFR_zeta_pvalue(asdf, subset_ttls, 
# 																		datasep, seg, stim_dur=20000, 
# 																		boolPlot=False, boolReturnRate=False)
					

# 					#which ones to process (only auditory responsive)

					


# 					nmult = len(multiplier)
# 					nsub_ttls = len(subset_ttls)
					
# 					ttlsperfile = 20000/tonedur
# 					nchanges = len(tonesdirlist)*ttlsperfile

# 					if nsub_ttls == len(tonesdirlist):
# 						new_sub_ttls = np.zeros(ttlsperfile*nsub_ttls)
# 						tt = 0
# 						for ttl in subset_ttls:
# 							for t in np.arange(ttlsperfile):
# 								new_sub_ttls[tt] = ttl+t*tondur
# 								tt+=1

# 					stims = 

# 					pattern, ttlarray = patternGen(asdf[neurons], subset_ttls, stims, num_stim, 
# 												   datasep[seg],  window=timewindow, force=False)

# 					def stacalc_sub(asdf:np.array, 
#                 					ttls:np.array):

# 						patnum = len(    ) / nreps;











# def stacalc_sub(asdf, ttls):
	

# #function [stas, stasigs, nSpikes, meanpic] = stacalc_sub(asdf)
#         patnum = length(    ) / nreps;
#         inittimes = ttls(1:patnum:end) - segst;
#         inittimes = [inittimes; segttls{seg}(end) + 25];
		
#         pat_ttls = {};
#         for i = 1:nreps
#             offset = patnum * (i - 1);
#             pat_ttls{i} = segttls{seg}((1:patnum) + offset) - segst - inittimes(i);
#         end
		
#         asdf_minis = cell(length(inittimes) - 1, 1);
#         for i = 1:(length(inittimes)-1)
#             asdf_minis{i} = ASDFChooseTime(asdf, inittimes(i), inittimes(i+1));
#         end
		
#         %% prepare STA picture
		
#         fullpic = zeros(tn, patnum, 2);
#         for j = 1:2
#             for i = 1:patnum
#                 fullpic(pt.picked_tones{i, j}, i, j) = 1;
#             end
#         end
#         %figure(33553);
#         %imagesc(fullpic);
#         picoffset = 9;
#         fullpic = [ones(tn, picoffset, 2)/2, fullpic, ones(tn, 9-picoffset, 2)/2];
#         partpic = zeros(tn, 10, patnum, 2);
#         for i = 1:patnum
#             partpic(:, :, i, :) = fullpic(:, i:i+9, :);
#         end
#         meanpic = mean(partpic, 3);
		
		
#         %% let's do analysis per neuron
#         %shortlist = [62 68 71 72 74 76 90 92 94 97 98 105 107 112 124 127 141 143 151 162 163 167];
		
#         %figure(5000 + seg);
#         %clf
		
#         % separate plot and calculation. Do calculation first.
#         stas = np.array((nNeu, 1), type='object');
#         stasigs = cell(nNeu, 1);
#         nSpikes = zeros(nNeu, 1);
#         tic;
#         for neu = 1:nNeu
#             fprintf('*');
#             if mod(neu, 50) == 0
#                 fprintf(' %d\n', neu);
#             end
#             % 1. get spike timing % done by asdf_minis
#             % 2. make correspondence with segment TTLs
#             hc = zeros(1, patnum);
#             for i = 1:nreps
#                 hc = hc + histcounts(asdf_minis{i}{neu}, [pat_ttls{i}; pat_ttls{i}(end) + 25]);
#             end
#             % 3. calculate a weighted average
#             fhc = find(hc);
#             fhcv = zeros(1, 1, length(fhc));
#             fhcv(:) = hc(fhc);
#             nSpike = sum(fhcv(:));
#             nSpikes(neu) = nSpike;
#             sizep = size(partpic);
#             staraw = sum(partpic(:, :, fhc, :) .* repmat(fhcv, [sizep(1), sizep(2), 1, 2]), 3);
#             stas{neu} = staraw / nSpike;
			
#             % get significance calculation. takes about a second per neuron.
#             stasigs{neu} = binocdf_unique(staraw, nSpike, meanpic);
#         end
#         fprintf(' done!\n');
#         toc
#     end
# end

# 					stacalc_sub()
# 					print('')	            
		   
