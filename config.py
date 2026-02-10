#!/usr/bin/env python3
'''
Config file required to run "auditory_analysis_pipeline.py"

Authors: Brian R. Mullen
Date: 2026-02-05

'''

import os
import numpy as np

class configs():
	def __init__(self):
		#stimdir, SAME FILES USED TO STIMULATE
		self.stimdir = '/Users/ackmanadmin/Documents/test_dataset_auditory_pipeline/python/Auditory/stimgen/'
		#sequence of data sequences, NEED TO SET
		# data segment, location of stimulus file, distribution fit, multiplier 
		self.stim_dict = {'seg1':['npx_gen1/fullfield_newspeakers/fullfield_newspeakers.txt', 'kent', [1, 10]],
						  # 'seg6':['npx_gen1/fullfield_newspeakers/fullfield_newspeakers.txt', 'kent', [10]],
						  
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
	    for keys, values in vars(self).items():
	    	if isinstance(values, dict):
	    		print('\t{}'.format(keys), file=f)
	    		for key, value in values.items():
	    			print('\t\t{0}:{1}'.format(key, value), file=f)
	    	else:
	        	print('\t{0}:{1}'.format(keys, values), file=f)