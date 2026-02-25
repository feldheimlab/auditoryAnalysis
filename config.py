#!/usr/bin/env python3
'''
Config file required to run "auditory_analysis_pipeline.py"

Authors: Brian R. Mullen
Date: 2026-02-05

'''

import os
import numpy as np

class configs():
	'''
	Centralized configuration for the auditory analysis pipeline.

	All tunable parameters for stimulus mappings, analysis windows,
	statistical tests, and significance thresholds are set here.
	Path attributes (dataloc, parentdir, savedir, imagedir) and run
	arguments (spikesorting, class_col, group, probe, rate) are
	populated by the pipeline script at runtime.

	Attributes
	----------
	stimdir : str
		Path to the directory containing stimulus files.
	stim_dict : dict
		Mapping of segment keys (e.g. 'seg5') to [stimulus_file, fit_type, multipliers].
		fit_type is 'kent' for spatial receptive fields or 'RandomChord' for STA analysis.
	spont_win : list of int
		[start, end] in ms for the spontaneous activity window used as baseline.
	windows : np.ndarray
		(nWin, 2) array of [start, end] time windows in ms for evoked activity.
	testdist : list of str
		Statistical tests to use: 'poisson', 'quasipoisson', and/or 'nbinom'.
	siglvl : list of float
		Significance thresholds corresponding to each test in testdist.
	timewindow : list of int
		[start, end] in ms for spike pattern generation.
	'''
	def __init__(self):
		#stimdir, SAME FILES USED TO STIMULATE
		self.stimdir = '../../GitHub/auditory_stim_files/python/Auditory/stimgen/gen5/'
		#sequence of data sequences, NEED TO SET
		# data segment, location of stimulus file, distribution fit, multiplier 
		self.stim_dict = {'seg1':['fullfield.txt', 'kent', [10]],
						  #'seg6':['npx_gen1/fullfield_newspeakers/fullfield_newspeakers.txt', 'kent', [10]],

						 #'seg2':['RandomChord/randomchord4810_newspeakers', 'RandomChord', [1]]
						 }

		#determine auditory responsive neurons, NEED TO SET

		self.spont_win = [700,1000]
		self.windows = np.array([[0,20]])#,[20,100],[100,500]]) #windows
		self.testdist = ['quasipoisson']#['poisson','quasipoisson', 'nbinom'] #tests
		self.siglvl = [0.001]#[0.001,0.001,0.001]#[0.001,0.01,0.05] #alpha significance
		self.nWin = self.windows.shape[0] #n of windows

		#pattern generation window
		self.timewindow = [0,1000] #pattern generation window
		
		#save locations, updated in script
		self.dataloc = None
		self.parentdir = None
		self.savedir = None
		self.imagedir = None

		#run arguments, updated in script
		self.spikesorting = None
		self.class_col = None
		self.group = None
		self.probe = None
		self.rate = None
		self.class_col = None
		self.classification = None

	def write_attributes(self, f):
		'''
		Write all config attributes to a file handle for logging.

		Parameters
		----------
		f : file object
			Open file handle to write to (used with print(..., file=f)).
		'''
		for keys, values in vars(self).items():
			if isinstance(values, dict):
				print('\t {}'.format(keys), file=f)
				for key, value in values.items():
					print('\t\t{0}: {1}'.format(key, value), file=f)
			else:
				print('\t{0}: {1}'.format(keys, values), file=f)