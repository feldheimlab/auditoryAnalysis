import os
import sys
import math

sys.path.append('./python')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_neurons_relative_to_probe(data_obj, save_image_dir):
		
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
			data_obj.cluster['group_c'] = data_obj.cluster[data_obj.class_col]

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