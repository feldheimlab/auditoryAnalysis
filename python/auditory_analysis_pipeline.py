import os
import sys
import math

sys.path.append('/home/feldheimlab/Documents/auditoryAnalysis/python')
sys.path.append('/home/feldheimlab/Documents/pyWholeBrain/')

from scipy.stats import poisson
from scipy.stats import norm
import scipy.io as sio

from preprocessing import *
from distributions_fit import *
from hdf5manager import hdf5manager as h5

from scipy.stats import poisson
from scipy.stats import nbinom

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == '__main__':

	import argparse
	import time
	import datetime

	# Argument Parsing
	# -----------------------------------------------
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--input_directory', type = str,
		required = True, 
		help = 'path to the vision files for processing')
	ap.add_argument('-idict', '--input_dict', type = str,
		required = False, #THIS WILL ULTIMATELY BE TRUE
		help = 'path to stimulus dictionary files, indicating what the data is and how it will be processed')
	ap.add_argument('-p', '--probe', type = str,
		default='npxl', 
		help = 'generates the correct probemap, this can be "A", "AN", or "npxl"')
	ap.add_argument('-f', '--fps', type = int,
		default=30000,  
		help = 'frames per second for data collection')
	ap.add_argument('-ds', '--dontsave', action='store_false',
		help = 'if flagged, it will not save output graphs and initial assessements of the data')
	ap.add_argument('-g', '--group_tsv', type = str,
		default='cluster_info.tsv',  
		help = 'tsv with specified classified clusters')
	ap.add_argument('-c', '--class_col', type = str,
		default='KSLabel', 
		help = 'column name of group tsv that indicates that classification of neuronal dataset')
	ap.add_argument('-class', '--class', type = str,
		default='all', 
		help = "which classifications to include: 'all', 'good', 'mua'")
	args = vars(ap.parse_args())


	#necessary paths
	visionloc = args['input_directory']
	assert os.path.exists(visionloc), 'Could not find: {}'.format(kilosortloc)
	parentdir = os.path.dirname(visionloc)
	kilosortloc = os.path.join(parentdir, 'kilosort4')
	savedir = os.path.join(parentdir, 'results')
	
	if not os.path.exists(savedir):
		os.mkdir(savedir)
	print('Processed dataset will be saved: ', savedir)

	probe = args['probe']
	rate = args['fps']
	clusterdef = args['group_tsv']
	class_col = args['class_col']
	classification = args['class']
	save = args['dontsave']

	# HARDCODDED PARAMETERS
	timewindow = [0,2000] #pattern generation window

	#determine auditory responsive neurons
	spont_win = [700,1000]
	windows = np.array([[5,20]])#,[20,100],[105,120]]) #windows
	testdist = ['poisson','quasipoisson', 'nbinom'] #tests
	siglvl = [0.001,0.001,0.001]#[0.001,0.01,0.05] #alpha significance
	nWin = windows.shape[0] #n of windows

	#MAKE THIS AN INPUT FILE FOR THE PIPELINE!
	stim_dict = {'stimdir':'/home/feldheimlab/python/Auditory/stimgen/npx_gen1/',
				'seg2':['fullfield_newspeakers/fullfield_newspeakers.txt', 'kent', [1,10]],
				'seg3':['RandomChord/RandomChord', 'RandomChord', [1]]}

	stimdir = stim_dict['stimdir']

	kilosort=True
	vision=False

	#paths for necessary data
	try:
		risedpath = os.path.join(visionloc, 'ttlTimes.mat')
		segmentlengthspath = os.path.join(visionloc, 'segmentlengths.mat')
		rised = np.squeeze(sio.loadmat(risedpath)['ttlTimes'])	#load TTLs
		datasep = np.squeeze(sio.loadmat(segmentlengthspath)['segmentseparations']) #data separations
	except:
		print('Could not find the ttlTimes and segmentlengths files in vision, looking in parent directory')
		risedpath = os.path.join(parentdir, 'ttlTimes.mat')
		segmentlengthspath = os.path.join(parentdir, 'segmentlengths.mat')
		rised = np.squeeze(sio.loadmat(risedpath)['ttlTimes'])	#load TTLs
		datasep = np.squeeze(sio.loadmat(segmentlengthspath)['segmentseparations']) #data separations

	basicinfopath = os.path.join(visionloc, 'xy.mat')
	asdfpath = os.path.join(visionloc,'asdf.mat')

	#locations of clusters calculated by Vision
	basicinfo = sio.loadmat(basicinfopath)	

	# add a zero for the start of the experiments
	datasep2 = np.zeros(datasep.shape[0]+1)
	datasep2[1:] = datasep
	datasep = datasep2

	#load channel maps
	chanposition = probeMap(probe=probe)

	#load asdf information
	asdf_raw = sio.loadmat(asdfpath)['asdf_raw']
	# aud_ids = sio.loadmat(os.path.join(dataloc, 'AuditorySpotSummary_1-posneu.mat'))['posneu'] #for comparison 
	
	#cluster information from kilosort
	cluster = pd.read_csv(os.path.join(kilosortloc, 'cluster_info.tsv'), sep='\t') #class 

	#simplify the asdf and determine fr
	nneur = asdf_raw.shape[0]-2
	asdf = np.empty((nneur), dtype='object')

	for n in np.arange(nneur):
		asdf[n] = np.squeeze(asdf_raw[n][0]) #simplifies the asdf

	if classification == 'all':
		nNeu = asdf.shape[0]#n of neurons
		neurons = np.arange(nNeu)
	else:
		neurons = cluster[cluster[class_col]==classification]
		nNeu = len(neurons)


	with open(savedir+'out.txt', 'w') as f:

		for key, value in stim_dict.items():
			if key == 'stimdir':
				continue
			#preprocessing
			print('New trial:', key, value, file=f)
			file = value[0]
			fit = value[1]
			multiplier = value[2]
			seg = int(key[-1:])

			if len(key) > 4:
				seg = int(key[-2:])

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
			print(stims)

			print('\nUnique indices: ', stim_ind.shape, file=f)
			print(stim_ind)

			#get the ttls of interest
			ttls = getTTLseg(seg=seg, ttls=rised, datasets=datasep)
			
			nmult = len(multiplier)
			nttls = len(ttls)

			for m, mult in enumerate(multiplier):

				if nmult == 1:
					print('Using all TTLs to determine pattern, based on 1 mutlipier', file=f)
					subset_ttls = ttls
				else:
					print('Using {0} TTLs to determine pattern, based on {1} mutlipiers'.format(nttls/nmult, nmult), file=f)
					ttlsmult = int(nttls/nmult)
					if m == 0:
						subset_ttls = ttls[:ttlsmult]
					if (m+1)==nmult:
						subset_ttls = ttls[-ttlsmult:]
					else:
						subset_ttls = ttls[m*ttlsmult:(m+1)*ttlsmult]

				#get the pattern of this dataset
				pattern, ttlarray = patternGen(asdf[neurons], subset_ttls, stims, num_stim, 
											   datasep[seg],  window=timewindow, force=False)


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



				# activity_df2, IFR, vecTime = sigAudFR_zeta_pvalue(asdf, subset_ttls, datasep, seg, stim_dur=100, boolPlot = False, boolReturnRate=False)
				# activity_df[activity_df2.columns] = activity_df2.values

				activity_df['cluster'] = neurons

				aud_poisson =  np.array([[None] * nWin] * nNeu)
				aud_qpoisson =  np.array([[None] * nWin] * nNeu)

				for w in range(nWin):
					aud_poisson[:,w] = (activity_df['poisson win {} neg'.format(str(w))] + 
										activity_df['poisson win {} pos'.format(str(w))]).values
					aud_qpoisson[:,w] = (activity_df['quasipoisson win {} neg'.format(str(w))] + 
										 activity_df['quasipoisson win {} pos'.format(str(w))]).values
				
				if save:
					seqsavedir = savedir + '{0} mult {1}/'.format(key, mult)
					if not os.path.exists(seqsavedir):
						os.mkdir(seqsavedir)
					np.save(os.path.join(seqsavedir, 'pattern.npy'), pattern)
					np.save(os.path.join(seqsavedir, 'good_neurons.npy'), neurons)   
					np.save(os.path.join(seqsavedir, 'aud_neurons_poisson.npy'), aud_poisson) 
					np.save(os.path.join(seqsavedir, 'aud_neurons_qpoisson.npy'), aud_qpoisson)     
					np.save(os.path.join(seqsavedir, 'spont_win'), spont_win)
					np.save(os.path.join(seqsavedir, 'windows'), windows)
					# np.save(os.path.join(seqsavedir, 'InstantaneousFR.npy'), IFR)
					# np.save(os.path.join(seqsavedir, 'InstantaneousVecTime.npy'), vecTime)
					activity_df.to_csv(os.path.join(seqsavedir, 'activity-df.csv'))
					
					print('Preprocessing data saved to :', seqsavedir, file=f)

				#Fitting the data, only process select neurons
				for w in range(nWin):
					neuron_assess = np.where(aud_qpoisson[:,w]==1)[0] # I USED THE QUASIPOISSON TO DETERMINE WHICH NEURONS TO FIT TO THE DISTRIBUTION

					if kilosort:
						neurons = cluster[cluster[class_col]=='good']
						neurons = pd.concat([neurons, cluster[cluster[class_col]=='mua']])

						groups = ['good', 'mua', 'noise']
						print('\n', file=f)
						if w == 0:
							for group in groups:
								frac = len(cluster[cluster[class_col]==group])/len(cluster)
								print('{0} fraction {1}/{2}: {3}%'.format(group, len(cluster[cluster[class_col]==group]),
																		  len(cluster), np.round(frac*100, 2)), file=f)
						if 'neuron_assess' in locals():
							print('\nIn window {} - {} ms'.format(windows[w,0],windows[w,1]), file=f)
							neuron_assess = np.where(aud_qpoisson[:,w]==1)[0]
							aud_resp = neuron_assess

							for group in groups:
								frac = len(neurons[neurons[class_col]==group])/len(neurons)
								print('{0} fraction {1}/{2}: {3}%'.format(group, len(neurons[neurons[class_col]==group]), 
																		 len(neurons), np.round(frac*100, 2)), file=f)
								if group == 'good':
									frac2 = len(aud_resp)/len(neurons[neurons[class_col]==group])
									print('\tgood and auditory responsive fraction {0}/{1}: {2}%'.format(len(aud_resp), len(neurons[neurons['group']==group]), 
																		  np.round(frac2*100, 2)), file=f)
						else:
							aud_resp = []

							for group in groups:
								frac = len(neurons[neurons[class_col]==group])/len(neurons)
								if group == 'good':
									frac2 = len(aud_resp)/len(neurons[neurons[class_col]==group])
									print('good and auditory responsive fraction {0}/{1}: {2}%'.format(len(aud_resp), len(neurons[neurons['group']==group]), 
																		  np.round(frac2*100, 2)))  
								print('{0} fraction {1}/{2}: {3}%'.format(group, len(neurons[neurons[class_col]==group]), 
																		 len(neurons), np.round(frac*100, 2)))
					elif vision:
						print('Processing vision file: ', dataloc, file=f)
						print('/n')
						neurons = np.arange(asdf.shape[0])
						if 'neuron_assess' in locals():
							print('\nIn window {} - {} ms'.format(windows[w,0],windows[w,1]), file=f)
							neuron_assess = np.where(aud_qpoisson[:,w]==1)[0]
							aud_resp = neuron_assess
							frac2 = len(aud_resp)/len(neurons)
							print('\tgood and auditory responsive fraction {0}/{1}: {2}%'.format(len(aud_resp), len(neurons[neurons[class_col]==group]), 
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

						print('\nFitting kent distribution')

						print('\tCalculating counts for the window size')
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
								fresidual = fitdist-mean_data
								
								kaic = aic_leastsquare(fresidual, p)
								kbic = bic_leastsquare(fresidual, p)

								kentres = chiSquaredTest(mean_data.reshape(mean_data.size), fitdist.reshape(fitdist.size), p)

								parameters[n] = p
								variances[n] = var
								modelfits[n] = fitdist
								datafits[n] = mean_data
								aic_bic[n] = [uaic, kaic, ubic, kbic]
								sumresid[n] = [np.sum(np.abs(uresidual)), resid_store[index]]

						if save:
							winsavedir = seqsavedir + '{0} fit win {1}/'.format(fit, w)
							if not os.path.exists(winsavedir):
								os.mkdir(winsavedir)
							np.save(os.path.join(winsavedir,'neuron_assess.npy'), neuron_assess)
							np.save(os.path.join(winsavedir, 'parameters.npy'), parameters)
							np.save(os.path.join(winsavedir, 'variances.npy'), variances)   
							np.save(os.path.join(winsavedir, 'modelfits.npy'), modelfits) 
							np.save(os.path.join(winsavedir, 'datafits.npy'), datafits)     
							np.save(os.path.join(winsavedir, 'aic_bic.npy'), aic)
							np.save(os.path.join(winsavedir, 'sumresid.npy'), sumresid)
							
							print('Kent distribution data saved to :', winsavedir, file=f)    
					elif fit == 'RandomChord':
						#code here
						print('')	            
			   
