#!/usr/bin/env python3
'''
Preprocessing of spike sorted data

Authors: Brian R Mullen
Date: 2023-06-01
'''

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zetapy import ifr, zetatest, zetatstest, zetatest2, zetatstest2, plotzeta, plottszeta, plotzeta2, plottszeta2

import scipy.io as sio
from scipy.stats import poisson
from scipy.stats import norm
from scipy.stats import nbinom


def ask_yes_or_no(msg: str) -> bool:
    while True:
        if (user := input(msg).lower()) in ('y', 'yes'):
            return True
        if user in ('n' ,'no'):
            return False
        print('Invalid input. Please try again...')


def probeMap(probe='AN'):

    '''
    Returns common probe maps

    Masmanadis (UCLA) probes: A are wide-spaced 4 shank probe (64 channels per shank/256 channels per probe; total spacing is 1200um)
                              AN are narrow-spaced 4 shank probe (64 channels per shank/256 channels per probe; total spacing is 600um)

    Neuropixel 2.0 4 shanke probes: The below map is 4 shanks taken at equal depths.  
    
    Arguments:
        probe: string indicating which probe map to return

    Returns:
        probeMap: channel locations of each predetermined configuration

    '''

    if (probe=='AN')|(probe=='A'):
        probe_map = np.array([[975,0],[875,0],[775,0],[675,0],[575,0],[475,0],[375,0],
                        [275,0],[175,0],[75,0],[0,0],[50,16],[100,20],[150,20],[200,20],
                        [250,20],[300,20],[1050,20],[1000,20],[950,20],[900,20],[850,20],
                        [800,20],[750,20],[700,20],[650,20],[600,20],[550,20],[500,20],
                        [450,20],[400,20],[350,20],[300,-20],[350,-20],[400,-20],[450,-20],
                        [500,-20],[550,-20],[600,-20],[650,-20],[700,-20],[750,-20],[800,-20],
                        [850,-20],[900,-20],[950,-20],[1000,-20],[1050,-20],[250,-20],[200,-20],
                        [150,-20],[100,-20],[50,-16],[25,0],[125,0],[225,0],[325,0],[425,0],
                        [525,0],[625,0],[725,0],[825,0],[925,0],[1025,0],[1025,200],[925,200],
                        [825,200],[725,200],[625,200],[525,200],[425,200],[325,200],[225,200],
                        [125,200],[25,200],[50,184],[100,180],[150,180],[200,180],[250,180],
                        [1050,180],[1000,180],[950,180],[900,180],[850,180],[800,180],[750,180],
                        [700,180],[650,180],[600,180],[550,180],[500,180],[450,180],[400,180],
                        [350,180],[300,180],[350,220],[400,220],[450,220],[500,220],[550,220],
                        [600,220],[650,220],[700,220],[750,220],[800,220],[850,220],[900,220],
                        [950,220],[1000,220],[1050,220],[300,220],[250,220],[200,220],[150,220],
                        [100,220],[50,216],[0,200],[75,200],[175,200],[275,200],[375,200],
                        [475,200],[575,200],[675,200],[775,200],[875,200],[975,200],[1025,400],
                        [925,400],[825,400],[725,400],[625,400],[525,400],[425,400],[325,400],
                        [225,400],[125,400],[25,400],[50,384],[100,380],[150,380],[200,380],
                        [250,380],[1050,380],[1000,380],[950,380],[900,380],[850,380],[800,380],
                        [750,380],[700,380],[650,380],[600,380],[550,380],[500,380],[450,380],
                        [400,380],[350,380],[300,380],[350,420],[400,420],[450,420],[500,420],
                        [550,420],[600,420],[650,420],[700,420],[750,420],[800,420],[850,420],
                        [900,420],[950,420],[1000,420],[1050,420],[300,420],[250,420],[200,420],
                        [150,420],[100,420],[50,416],[0,400],[75,400],[175,400],[275,400],
                        [375,400],[475,400],[575,400],[675,400],[775,400],[875,400],[975,400],
                        [975,600],[875,600],[775,600],[675,600],[575,600],[475,600],[375,600],
                        [275,600],[175,600],[75,600],[0,600],[50,616],[100,620],[150,620],
                        [200,620],[250,620],[300,620],[1050,620],[1000,620],[950,620],[900,620],
                        [850,620],[800,620],[750,620],[700,620],[650,620],[600,620],[550,620],
                        [500,620],[450,620],[400,620],[350,620],[300,580],[350,580],[400,580],
                        [450,580],[500,580],[550,580],[600,580],[650,580],[700,580],[750,580],
                        [800,580],[850,580],[900,580],[950,580],[1000,580],[1050,580],[250,580],
                        [200,580],[150,580],[100,580],[50,584],[25,600],[125,600],[225,600],
                        [325,600],[425,600],[525,600],[625,600],[725,600],[825,600],[925,600],
                        [1025,600]])
        if probe=='A':
            for i in range(4):
                probe_map[64*(i):(64*(i+1)),1]+=(i*200)

    if probe=='npxl':
        npmap = np.array([['NP2014',4,250,70],[0,27,0,1],[0,59,0,1],[0,27,15,1],[0,59,15,1],[0,27,30,1],[0,59,30,1],[0,27,45,1],
                [0,59,45,1],[0,27,60,1],[0,59,60,1],[0,27,75,1],[0,59,75,1],[0,27,90,1],[0,59,90,1],[0,27,105,1],
                [0,59,105,1],[0,27,120,1],[0,59,120,1],[0,27,135,1],[0,59,135,1],[0,27,150,1],[0,59,150,1],[0,27,165,1],
                [0,59,165,1],[0,27,180,1],[0,59,180,1],[0,27,195,1],[0,59,195,1],[0,27,210,1],[0,59,210,1],[0,27,225,1],
                [0,59,225,1],[0,27,240,1],[0,59,240,1],[0,27,255,1],[0,59,255,1],[0,27,270,1],[0,59,270,1],[0,27,285,1],
                [0,59,285,1],[0,27,300,1],[0,59,300,1],[0,27,315,1],[0,59,315,1],[0,27,330,1],[0,59,330,1],[0,27,345,1],
                [0,59,345,1],[1,27,0,1],[1,59,0,1],[1,27,15,1],[1,59,15,1],[1,27,30,1],[1,59,30,1],[1,27,45,1],[1,59,45,1],
                [1,27,60,1],[1,59,60,1],[1,27,75,1],[1,59,75,1],[1,27,90,1],[1,59,90,1],[1,27,105,1],[1,59,105,1],
                [1,27,120,1],[1,59,120,1],[1,27,135,1],[1,59,135,1],[1,27,150,1],[1,59,150,1],[1,27,165,1],[1,59,165,1],
                [1,27,180,1],[1,59,180,1],[1,27,195,1],[1,59,195,1],[1,27,210,1],[1,59,210,1],[1,27,225,1],[1,59,225,1],
                [1,27,240,1],[1,59,240,1],[1,27,255,1],[1,59,255,1],[1,27,270,1],[1,59,270,1],[1,27,285,1],[1,59,285,1],
                [1,27,300,1],[1,59,300,1],[1,27,315,1],[1,59,315,1],[1,27,330,1],[1,59,330,1],[1,27,345,1],[1,59,345,1],
                [0,27,360,1],[0,59,360,1],[0,27,375,1],[0,59,375,1],[0,27,390,1],[0,59,390,1],[0,27,405,1],[0,59,405,1],
                [0,27,420,1],[0,59,420,1],[0,27,435,1],[0,59,435,1],[0,27,450,1],[0,59,450,1],[0,27,465,1],[0,59,465,1],
                [0,27,480,1],[0,59,480,1],[0,27,495,1],[0,59,495,1],[0,27,510,1],[0,59,510,1],[0,27,525,1],[0,59,525,1],
                [0,27,540,1],[0,59,540,1],[0,27,555,1],[0,59,555,1],[0,27,570,1],[0,59,570,1],[0,27,585,1],[0,59,585,1],
                [0,27,600,1],[0,59,600,1],[0,27,615,1],[0,59,615,1],[0,27,630,1],[0,59,630,1],[0,27,645,1],[0,59,645,1],
                [0,27,660,1],[0,59,660,1],[0,27,675,1],[0,59,675,1],[0,27,690,1],[0,59,690,1],[0,27,705,1],[0,59,705,1],
                [1,27,360,1],[1,59,360,1],[1,27,375,1],[1,59,375,1],[1,27,390,1],[1,59,390,1],[1,27,405,1],[1,59,405,1],
                [1,27,420,1],[1,59,420,1],[1,27,435,1],[1,59,435,1],[1,27,450,1],[1,59,450,1],[1,27,465,1],[1,59,465,1],
                [1,27,480,1],[1,59,480,1],[1,27,495,1],[1,59,495,1],[1,27,510,1],[1,59,510,1],[1,27,525,1],[1,59,525,1],
                [1,27,540,1],[1,59,540,1],[1,27,555,1],[1,59,555,1],[1,27,570,1],[1,59,570,1],[1,27,585,1],[1,59,585,1],
                [1,27,600,1],[1,59,600,1],[1,27,615,1],[1,59,615,1],[1,27,630,1],[1,59,630,1],[1,27,645,1],[1,59,645,1],
                [1,27,660,1],[1,59,660,1],[1,27,675,1],[1,59,675,1],[1,27,690,1],[1,59,690,1],[1,27,705,1],[1,59,705,1],
                [2,27,0,1],[2,59,0,1],[2,27,15,1],[2,59,15,1],[2,27,30,1],[2,59,30,1],[2,27,45,1],[2,59,45,1],[2,27,60,1],
                [2,59,60,1],[2,27,75,1],[2,59,75,1],[2,27,90,1],[2,59,90,1],[2,27,105,1],[2,59,105,1],[2,27,120,1],[2,59,120,1],
                [2,27,135,1],[2,59,135,1],[2,27,150,1],[2,59,150,1],[2,27,165,1],[2,59,165,1],[2,27,180,1],[2,59,180,1],[2,27,195,1],
                [2,59,195,1],[2,27,210,1],[2,59,210,1],[2,27,225,1],[2,59,225,1],[2,27,240,1],[2,59,240,1],[2,27,255,1],[2,59,255,1],
                [2,27,270,1],[2,59,270,1],[2,27,285,1],[2,59,285,1],[2,27,300,1],[2,59,300,1],[2,27,315,1],[2,59,315,1],[2,27,330,1],
                [2,59,330,1],[2,27,345,1],[2,59,345,1],[3,27,0,1],[3,59,0,1],[3,27,15,1],[3,59,15,1],[3,27,30,1],[3,59,30,1],
                [3,27,45,1],[3,59,45,1],[3,27,60,1],[3,59,60,1],[3,27,75,1],[3,59,75,1],[3,27,90,1],[3,59,90,1],[3,27,105,1],
                [3,59,105,1],[3,27,120,1],[3,59,120,1],[3,27,135,1],[3,59,135,1],[3,27,150,1],[3,59,150,1],[3,27,165,1],[3,59,165,1],
                [3,27,180,1],[3,59,180,1],[3,27,195,1],[3,59,195,1],[3,27,210,1],[3,59,210,1],[3,27,225,1],[3,59,225,1],[3,27,240,1],
                [3,59,240,1],[3,27,255,1],[3,59,255,1],[3,27,270,1],[3,59,270,1],[3,27,285,1],[3,59,285,1],[3,27,300,1],[3,59,300,1],
                [3,27,315,1],[3,59,315,1],[3,27,330,1],[3,59,330,1],[3,27,345,1],[3,59,345,1],[2,27,360,1],[2,59,360,1],[2,27,375,1],
                [2,59,375,1],[2,27,390,1],[2,59,390,1],[2,27,405,1],[2,59,405,1],[2,27,420,1],[2,59,420,1],[2,27,435,1],[2,59,435,1],
                [2,27,450,1],[2,59,450,1],[2,27,465,1],[2,59,465,1],[2,27,480,1],[2,59,480,1],[2,27,495,1],[2,59,495,1],[2,27,510,1],
                [2,59,510,1],[2,27,525,1],[2,59,525,1],[2,27,540,1],[2,59,540,1],[2,27,555,1],[2,59,555,1],[2,27,570,1],[2,59,570,1],
                [2,27,585,1],[2,59,585,1],[2,27,600,1],[2,59,600,1],[2,27,615,1],[2,59,615,1],[2,27,630,1],[2,59,630,1],[2,27,645,1],
                [2,59,645,1],[2,27,660,1],[2,59,660,1],[2,27,675,1],[2,59,675,1],[2,27,690,1],[2,59,690,1],[2,27,705,1],[2,59,705,1],
                [3,27,360,1],[3,59,360,1],[3,27,375,1],[3,59,375,1],[3,27,390,1],[3,59,390,1],[3,27,405,1],[3,59,405,1],[3,27,420,1],
                [3,59,420,1],[3,27,435,1],[3,59,435,1],[3,27,450,1],[3,59,450,1],[3,27,465,1],[3,59,465,1],[3,27,480,1],[3,59,480,1],
                [3,27,495,1],[3,59,495,1],[3,27,510,1],[3,59,510,1],[3,27,525,1],[3,59,525,1],[3,27,540,1],[3,59,540,1],[3,27,555,1],
                [3,59,555,1],[3,27,570,1],[3,59,570,1],[3,27,585,1],[3,59,585,1],[3,27,600,1],[3,59,600,1],[3,27,615,1],[3,59,615,1],
                [3,27,630,1],[3,59,630,1],[3,27,645,1],[3,59,645,1],[3,27,660,1],[3,59,660,1],[3,27,675,1],[3,59,675,1],[3,27,690,1],
                [3,59,690,1],[3,27,705,1],[3,59,705,1]])
        
        probe_map = npmap[1:,1:3].copy()
        probe_map[:,0] = npmap[1:,2]
        probe_map[:,1] = (npmap[1:,0].astype('float')*250)+ npmap[1:,1].astype('float')

    return np.array(probe_map).astype('float')


def ttl_rise(digital_data, rate=20000):
    '''
    Returns the rise times of each TTL from the digital waveform
    
    Arguments:
        digital_data: digital waveform from digital channel from recording
        rate: recording rate of the digital channel

    Returns:
       ttl array: 1-D array fo the rising times of each ttl found in the digital data
    '''


    digital_data = np.squeeze(digital_data)
    dif = digital_data[:-1]-digital_data[1:]
    rise = np.where(dif<-0.5)[0]
    rise_d = []
    #there
    for r, ris in enumerate(rise):
        if r == 0:
            rise_d.append(ris)
        #if the ttl happens within 9.5ms of last ttl, exclude the ttl
        elif (ris) >= (rise_d[-1]+(.0095*rate)): 
            rise_d.append(ris)
            
    return np.array(rise_d)*1000/rate#per ms


def readStimFile(wd, file):
    '''
    Reads the stimulation files formated for this analysis. 
    This file is generated at the time of generating the sound stimulations.  
    
    Arguments:
        wd: working directory where the file is saved
        file: the stimulation file, indicating when each stimulation occured

    Returns:
       stims: 2-D array [order of one set of trials, N-trials]
       num_stim: list of lists, indicating the number of stimulations for each condition 
                ie:[[azim stims], [elev stims], [laser stims]]
       stim_ind: 1-D of the unique indices of each the stimulation
    '''


    lines = []
    n = 0
    with open(os.path.join(wd, file)) as f:
        [lines.append(line.strip()) for line in f.readlines()]
    f.close()
    
    print('Reading file: \n\t', os.path.join(wd, file))
    #get the different stimulations
    
    num_stim1 = lines[2] #stimulation list 1
    num_stim2 = lines[3] #stimulation list 2
    num_stim3 = lines[4] #stimulation list 3
    
    num_stim1 = num_stim1.split(' ')
    num_stim2 = num_stim2.split(' ')
    num_stim3 = num_stim3.split(' ')
    
    if not num_stim2==['']: # if its not empty
        num_stim = [num_stim1, num_stim2]
    
    if not num_stim3==['']: # if its not empty
        num_stim.append(num_stim3)
#     print(num_stim.shape)

    #get the order of stimulations
    for l, line in enumerate(lines[5:]):
        if l == 0:
            first = []
            n0 = 0
            for n, li in enumerate(line):
                if li == ' ':
                    try:
                        first.append(int(line[n0:n]))
                    except Exception as e: e
                    n0=n
            first.append(int(line[n0:]))
            stims = np.zeros((len(lines[4:]), len(first)))*np.nan
            stims[0] = np.array(first) 
        else:
            n0 = 0
            i = 0
            for n, li in enumerate(line):
                if li == ' ':
                    try:
                        stims[l, i] = int(line[n0:n])
                        i+=1
                    except Exception as e: e
                    n0=n      
            stims[l,i] = int(line[n0:])

    stim_ind = np.unique(stims)

    if np.isnan(stim_ind).any():
        n_stim = len(stim_ind)-1
        n_trial = np.sum(stims == 1)
        stims = stims[~np.isnan(stims)].reshape((n_trial, n_stim))
        
    stim_ind = np.unique(stims)             
            
    return stims, num_stim, stim_ind


def PatternToCount(pattern, timerange, timeBinSz = 10, verbose=False):

    '''
    Convert a spike pattern array into firing rate histograms by binning
    spikes in time.

    Parameters
    ----------
    pattern : np.ndarray, dtype=object
        Spike pattern array of 3-5 dimensions. For a 5D array the axes are
        [neurons, elevation, azimuth, condition, trial], where each element
        contains an array of spike times.
    timerange : list or int
        Time window for binning. If a two-element list, interpreted as
        [start, end] in ms. If a single int or one-element list, the window
        is [0, timerange] ms. If None, defaults to [0, 2000] ms.
    timeBinSz : int, optional
        Size of each time bin in ms. Default is 10.
    verbose : bool, optional
        If True, print additional information during processing.

    Returns
    -------
    fr : np.ndarray
        Binned spike counts. Has the same leading dimensions as ``pattern``
        with an additional final axis of length ``len(bins) - 1``
        representing time bins.
    bins : np.ndarray
        Bin edges used for histogramming, in ms.
    '''



    # make 4D matrix of firing rates based on the events
    # visualize the firing rate based on the data given
    if timerange is None:
        maxtime = 2000
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
    
    bins = np.arange(mintime, maxtime+timeBinSz, timeBinSz)
    
    ns = pattern.shape

#     assert len(ns)<5, 'Give only patterns of shape (n neurons, nx, nrep) or (n neurons, nx, ny, nrep)'
    
    if len(ns) == 5:
        fr = np.zeros((ns[0], ns[1], ns[2], ns[3], ns[4], int(bins.shape[0]-1)))
        for n in np.arange(ns[0]):
            for x in np.arange(ns[1]):
                for y in np.arange(ns[2]):
                    for l in np.arange(ns[3]):
                        for t in np.arange(ns[4]):
                            try:
                                xs = np.squeeze(pattern[n,x,y,l,t])
                                fr[n,x,y,l,t], _ = np.histogram(xs, bins=bins)
                            except:
                                fr[n,x,y,l,t] = np.zeros(int(bins.shape[0]-1))
    
    elif len(ns) ==4:
        fr = np.zeros((ns[0], ns[1], ns[2], ns[3], int(bins.shape[0]-1)))
        for n in np.arange(ns[0]):
            for x in np.arange(ns[1]):
                for y in np.arange(ns[2]):
                    for t in np.arange(ns[3]):
                        try:
                            xs = np.squeeze(pattern[n,x,y,t])
                            fr[n,x,y,t], _ = np.histogram(xs, bins=bins)
                        except:
                            fr[n,x,y,t] = np.zeros(int(bins.shape[0]-1))
    
    elif len(ns) ==3:
        fr = np.zeros((ns[0], ns[1], ns[2], int(bins.shape[0]-1)))
        for n in np.arange(ns[0]):
            for x in np.arange(ns[1]):
                for t in np.arange(ns[2]):
                    try:
                        xs = np.squeeze(pattern[n,x,y,t])
                        fr[n,x,t], _ = np.histogram(xs, bins=bins)
                    except:
                        fr[n,x,t] = np.zeros(int(bins.shape[0]-1))
                        
    else:
        assert 'Unknown pattern shape'
    return fr, bins


def getTTLseg(seg, ttls, datasets):
    '''
    Extract TTL times that fall within a given data segment.

    Parameters
    ----------
    seg : int
        Segment index. TTLs are selected between ``datasets[seg]`` and
        ``datasets[seg+1]``.
    ttls : np.ndarray
        1-D array of all TTL times (in ms).
    datasets : np.ndarray
        1-D array of segment boundary timestamps. Must contain at least
        ``seg + 2`` elements.

    Returns
    -------
    np.ndarray
        TTL times that fall strictly within the segment boundaries
        ``(datasets[seg], datasets[seg+1])``.
    '''
    return ttls[(ttls<datasets[seg+1])&(ttls>datasets[seg])]


def PatternRaster3d(pattern3d, timerange=None, savepath=None):
    '''
    Create and save a raster plot from a 3D spike pattern array.

    Generates a grid of raster subplots where rows correspond to
    elevations, columns to azimuths, and scatter points within each
    subplot represent spike times across trials.

    Parameters
    ----------
    pattern3d : list or array-like
        3-D nested structure ``[elevation][azimuth][trial]`` where each
        innermost element is an array of spike times (in ms).
    timerange : list or None, optional
        Time window for the x-axis as ``[start, end]`` in ms. If a
        single int or one-element list, interpreted as ``[0, timerange]``.
        If None, defaults to ``[0, 1000]`` ms.
    savepath : str or None, optional
        File path to save the figure. If None, the figure is still
        displayed but not saved.
    '''
    # make/visualize a raster plot based on the data given
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
                    axs[x][y].scatter(xs, ys, s=1, marker='|', c='k')
                    xs = np.nan
                axs[x][y].set_yticks([])
                axs[x][y].set_xticks([])
                axs[x][y].set_xlim([mintime, maxtime])
                axs[x][y].set_ylim([-1, ns[2]+1])
        axs[int(ns[0]-1)][0].set_yticks([0, ns[2]])
        axs[int(ns[0]-1)][0].set_xticks([mintime, maxtime])
        axs[int(ns[0]-1)][0].set_xticklabels([0, dur])
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
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(savepath, dpi=300)
        plt.show()


def patternGen(asdf, ttls, stims, num_stim, ttl_trig,  window=0, force=False):
    '''
    Generate a 5D spike pattern array aligned to stimulus onsets.

    Assigns each neuron's spikes to the corresponding stimulus position
    based on TTL timing and the stimulus presentation order. When the
    number of TTLs does not match the expected count from the stimulus
    file, the ``force`` flag allows approximate recovery of missing TTLs.

    Parameters
    ----------
    asdf : np.ndarray, dtype=object
        Array of spike times per neuron (ASDF format). Each element is a
        1-D array of spike times in ms.
    ttls : np.ndarray
        1-D array of TTL onset times (in ms) for the current data
        segment.
    stims : np.ndarray
        2-D array of stimulus presentation order with shape
        ``(n_trials, stim_per_trial)``. Each entry is a stimulus index.
    num_stim : list of lists
        Unique stimulus values per dimension, e.g.
        ``[[elev_values], [azim_values]]`` or with a third laser
        dimension. The lengths determine the shape of the output pattern.
    ttl_trig : float
        Segment start time offset (in ms), used for diagnostic printing.
    window : list, optional
        Two-element list ``[start, end]`` in ms defining the peri-stimulus
        time window for extracting spikes. Default is ``[0, 2000]``.
    force : bool, optional
        If True, approximate missing TTLs when the TTL count does not
        match the expected number of stimulus presentations. Default is
        False.

    Returns
    -------
    pattern : np.ndarray, dtype=object
        5-D array ``[neurons, elevation, azimuth, condition, trials]``
        where each element contains an array of spike times relative to
        stimulus onset.
    ttlarray : np.ndarray
        TTLs reshaped into the same ``(n_trials, stim_per_trial)`` layout
        as ``stims``.
    '''
    if window == 0:
        window=[0,2000]
    
    stim_indices = np.unique(stims)
    
    n_trial, n_stim = stims.shape
    print('\nTrials: ', n_trial, '\nStimulations: ', n_stim)

    n_neurons = asdf.shape[0]
    n_ttls = ttls.shape[0]
    print('Nuerons:', n_neurons)
    if n_stim != len(num_stim):
        sz = []
        for stim in num_stim:
            sz.append(len(stim))
        n_diff_stim = len(num_stim)
    else:
        n_diff_stim = 1
        sz = 1

    print('TTLs found: ', n_ttls, '\nPresentations in stimfile: ', n_trial*n_stim)
    if n_ttls != n_trial*n_stim:
        print('\nNumber of stimulations do not match the number of ttls captured')
        if ask_yes_or_no('Would you like to try to force the analysis? [Y/N]: '):
            force = True
        assert force, 'If you would like to proceed, assumptions of some stimuli will occur. Use the force parameter to continue.'
        print('\t{} seconds of recording'.format(np.round(ttls[-1]-ttl_trig, 2)))
        print('\tFORCED APPROX OF TTLS WILL HAPPEN!')
    
    #Make ttl array in the same format as the stims array (this step makes indexin easy)
    ttlarray = np.zeros((n_trial, n_stim))
    lastttl = ttls[0]
    stim = 0
    trial = 0
    i =0
    
    for t, ttl in enumerate(ttls):
        if force:
            if ((ttl-lastttl) > 1500):#THIS IS ASSUMING LAGER THAN 1.5sec WILL BE APPROXIMATED! (ONLY WHEN FORCED)
                i+=1
                ttlarray[trial, stim] = lastttl+1000
                stim = (stim+1) % n_stim
                if stim == 0:
                    trial = (trial+1) % n_trial
               
                if ((ttl-lastttl) > 2500):
                    ttlarray[trial, stim] = lastttl+2000
                    stim = (stim+1) % n_stim
                    if stim == 0:
                        trial = (trial+1) % n_trial

                assert ((ttl-lastttl) < 3500), 'Too complicated a fix: {0} - {1} = {2}'.format(ttl, lastttl, ttl-lastttl) 
                        
                ttlarray[trial, stim] = ttl
            else:
                ttlarray[trial, stim] = ttl
        else:
            ttlarray[trial, stim] = ttl
        stim = (stim+1) % n_stim
        if stim == 0:
            trial = (trial+1) % n_trial
        lastttl = ttl
        
    if force:
        print('\tUpdated TTLs: ', np.sum(ttlarray!=0))
    
    print('\n')
    if n_diff_stim==3:
        pattern = np.empty((n_neurons,sz[0],sz[1],sz[2],n_trial), dtype=object)
        
        for n in np.arange(n_neurons):
            if (n % 100)==0:
                print('\tWorking on neuron ', n)
            neuron = np.squeeze(asdf[n])
            if np.isnan(neuron[0]):
                pattern[n] = np.nan
            else:
                for stim in stim_indices:
                    currentstim = np.sort(ttlarray[stims == int(stim)])
                    e, a, l = np.unravel_index(int(stim), sz)
                    for t, trial in enumerate(currentstim):
                         pattern[n,e,a,l,t] = neuron[(neuron>=trial+window[0])&(neuron<(trial+window[1]))]-trial
        
    if n_diff_stim==2:
        pattern = np.empty((n_neurons,sz[0], sz[1], 1, n_trial), dtype=object)
        
        for n in np.arange(n_neurons):
            if (n % 100)==0:
                print('\tWorking on neuron ', n)
            neuron = np.squeeze(asdf[n])
            for stim in stim_indices:
                currentstim = np.sort(ttlarray[stims == int(stim)])
                e, a = np.unravel_index(int(stim), sz)
                for t, trial in enumerate(currentstim):
                    pattern[n,e,a,0,t] = neuron[(neuron>=trial+window[0])&(neuron<(trial+window[1]))]-trial

    if n_diff_stim==1:
        pattern = np.empty((n_neurons, 1, n_stim, 1, n_trial), dtype=object) 
        
        for n in np.arange(n_neurons):
            if (n % 100)==0:
                print('\tWorking on neuron ', n)
            neuron = np.squeeze(asdf[n])
            for s, stim in enumerate(stim_indices):
                currentstim = np.sort(ttlarray[stims == int(stim)])
                for t, trial in enumerate(currentstim):
                    pattern[n,0,s,0,t] = neuron[(neuron>=trial+window[0])&(neuron<(trial+window[1]))]-trial
                    
    print('shape: ', pattern.shape)
    
    return pattern, ttlarray


def sigAudFR_zeta_pvalue(asdf:np.array, 
                        ttls:np.array, 
                        datasep:np.array, 
                        seg:int, 
                        stim_dur:int =10, 
                        boolPlot:bool = False, # Do we want to plot the results?
                        boolReturnRate:bool = False): # Do we want to return the instantaneous rates?  This increases the computational time emessly
    '''
    Test auditory responsiveness using the ZETA test for each neuron.

    Applies the ZETA test (Montijn et al.) to determine whether each
    neuron's firing pattern is significantly modulated by stimulus
    presentation. Optionally returns instantaneous firing rates.

    Parameters
    ----------
    asdf : np.ndarray, dtype=object
        Array of spike times per neuron (ASDF format). Each element is a
        1-D array of spike times in ms.
    ttls : np.ndarray
        1-D array of stimulus onset times (in ms).
    datasep : np.ndarray
        1-D array of segment boundary timestamps used to restrict each
        neuron's spikes to the relevant recording segment.
    seg : int
        Segment index. Spikes are selected between ``datasep[seg]`` and
        ``datasep[seg+1]``.
    stim_dur : int, optional
        Stimulus duration in ms, used to construct onset/offset event
        arrays for the t-test. Default is 10.
    boolPlot : bool, optional
        If True, plot the ZETA test results for each neuron. Default is
        False.
    boolReturnRate : bool, optional
        If True, return instantaneous firing rates (IFR) and
        corresponding time vectors. Increases computation time
        substantially. Default is False.

    Returns
    -------
    activity_df : pd.DataFrame
        DataFrame with one row per neuron containing columns ``'zeta p'``
        (ZETA test p-value) and ``'t test p'`` (mean-rate t-test
        p-value).
    IFR : np.ndarray or np.nan
        Array of instantaneous firing rate vectors per neuron if
        ``boolReturnRate`` is True, otherwise ``np.nan``.
    vecTime : np.ndarray or np.nan
        Array of time vectors corresponding to each neuron's IFR if
        ``boolReturnRate`` is True, otherwise ``np.nan``.
    '''
    print('Calculating Zeta p-values')
    # use minimum of trial-to-trial durations as analysis window size
    dblUseMaxDur = np.min(np.diff(ttls))

    # 50 random resamplings should give us a good enough idea if this cell is responsive.
    # If the p-value is close to 0.05, we should increase this number.
    intResampNum = 50 

    # what size of jittering do we want? (multiple of dblUseMaxDur; default is 2.0)
    dblJitterSize = 2.0

    # do we want to restrict the peak detection to for example the time during stimulus?
    # Then put (0, 1) here.
    tplRestrictRange = (0, np.inf)

    # create a T by 2 array with stimulus onsets and offsets so we can also compute the t-test
    arrEventTimes = np.transpose(np.array([ttls, ttls+stim_dur]))
    
    nneur = asdf.shape[0]
    for n, neuron in enumerate(asdf):
        # if n%100==0:
        neur = neuron[(neuron>datasep[seg])&(neuron<datasep[seg+1])]
        print('\tWorking on neuron ', n, neur.shape)
        dblZetaP, dZETA, dRate = zetatest(neur, arrEventTimes,
                                                    dblUseMaxDur=dblUseMaxDur,
                                                    intResampNum=intResampNum,
                                                    dblJitterSize=dblJitterSize,
                                                    boolPlot=boolPlot,
                                                    tplRestrictRange=tplRestrictRange,
                                                    boolReturnRate=boolReturnRate)
        if n == 0:
            activity_df = pd.DataFrame()
            if boolReturnRate:
                vecTime = np.empty((nneur), dtype='object')
                IFR = np.empty((nneur), dtype='object')
        if boolReturnRate:
            vecTime[n] = dRate['vecT']
            IFR[n] = dRate['vecRate']
        activity_df.loc[n, 'zeta p'] = dblZetaP
        activity_df.loc[n, 't test p'] = dZETA['dblMeanP']
    if boolReturnRate:
        return activity_df, IFR, vecTime
    else:
        return activity_df, np.nan, np.nan


def sigAudFRCompareSpont(pattern, spont_win, windows, test='poisson', siglvl=0.001, minspike=10):
    '''
    Identify auditory-responsive neurons by comparing evoked firing rates
    to a spontaneous baseline.

    For each analysis window, computes firing rates and tests whether
    they differ significantly from the spontaneous window using one or
    more statistical models. Supports Poisson, quasi-Poisson, and
    negative binomial tests.

    Parameters
    ----------
    pattern : np.ndarray, dtype=object
        4-D spike pattern array ``[neurons, x, y, trials]`` where each
        element contains spike times.
    spont_win : list
        Two-element list ``[start, end]`` in ms defining the spontaneous
        (baseline) time window.
    windows : np.ndarray
        Array of shape ``(nWin, 2)`` where each row is an evoked time
        window ``[start, end]`` in ms to test against spontaneous
        activity.
    test : str or list of str, optional
        Statistical test(s) to apply. Valid values are ``'poisson'``,
        ``'quasipoisson'``, and ``'nbinom'``. Default is ``'poisson'``.
    siglvl : float or list of float, optional
        Significance threshold(s) corresponding to each test. Default is
        0.001.
    minspike : int, optional
        Minimum spike count threshold for a neuron to be considered.
        Default is 10.

    Returns
    -------
    activity_df : pd.DataFrame
        DataFrame with one row per neuron containing columns for mean
        firing rate, variance, p-values, and boolean significance flags
        (positive and negative) for each window and test combination.
    '''

    if type(test) is str:
        test = [test]
    if type(siglvl) is float:
        siglvls = np.zeros(len(test))*siglvl
        siglvls = siglvls.tolist()
    if type(siglvl) is list:
        siglvls = siglvl
        
    assert len(test)==len(siglvls), 'Significance levels and number of testing distributions not equal'
    
    sz = pattern.shape
    nNeu = sz[0]
    nSeg = sz[1]*sz[2]*sz[3]
    nWin = windows.shape[0]
    
    columns = []
    
    if nWin > 1:
        for w in np.arange(nWin):
            ave = 'avg window {0} - {1} ms'.format(windows[w,0], windows[w,1])
            var = 'var window {0} - {1} ms'.format(windows[w,0], windows[w,1])
            columns.extend([ave, var])
    else:
        ave = 'avg window {0} - {1} ms'.format(windows[0,0], windows[0,1])
        var = 'var window {0} - {1} ms'.format(windows[0,0], windows[0,1])
        columns.extend([ave, var])
    
    spave = 'avg window {0} - {1} ms'.format(spont_win[0], spont_win[1])
    spvar = 'var window {0} - {1} ms'.format(spont_win[0], spont_win[1])
    columns.extend([spave, spvar])

    activity_df = pd.DataFrame(columns=columns)

    pvals = np.zeros((nNeu, nWin))
    spont_fr = np.zeros(nNeu)
    dur = np.diff(spont_win)[0]
    spont_fr, _ = PatternToCount(pattern=pattern, timerange=list(spont_win), timeBinSz=dur)
    spont_fr=spont_fr*1000/dur
    ns = spont_fr.shape
    activity_df[spave] = np.mean(spont_fr, axis=tuple(range(1, spont_fr.ndim)))
    activity_df[spvar] = np.var(spont_fr, axis=tuple(range(1, spont_fr.ndim)))
    
    activ_fr = np.zeros((nNeu, nWin, 2))
    tested = False
    for w in np.arange(nWin):
        window = windows[w]
        ave = 'avg window {0} - {1} ms'.format(windows[w,0], windows[w,1])
        var = 'var window {0} - {1} ms'.format(windows[w,0], windows[w,1])
        dur = np.diff(window)[0]

        fr, _ = PatternToCount(pattern=pattern,timerange=list(window), timeBinSz=dur)
        fr = fr*1000/dur
        ns = spont_fr.shape
        activity_df[ave] = np.mean(fr, axis=tuple(range(1, spont_fr.ndim)))
        activity_df[var] = np.var(fr, axis=tuple(range(1, spont_fr.ndim)))
        
        if 'poisson' in test:
            tested = True
            ind = test.index('poisson')
            pval = 'poisson pval compare win {0} - {1}'.format(str(w), nWin)
            activity_df[pval] = poisson.cdf(activity_df[ave]*dur*nSeg/1000, mu=activity_df[spave]*dur*nSeg/1000)
                        
            pospval = 1 - activity_df[pval]
            
            possig = pospval <= siglvls[ind] #positive significance
            negsig = activity_df[pval] <= siglvls[ind] #negative significance
            
            fr_limit = activity_df[ave]*dur/1000 >= minspike
#             possig[fr_limit] = False 
#             negsig[fr_limit] = False
            
            possig[activity_df[var]==0]=False
            negsig[activity_df[var]==0]=False

            print('Window {0}: Identified significant reposive neurons that {1} increased and {2} decreased firing rates out of {3} with {4} test.'.format(str(w), np.sum(possig), np.sum(negsig),len(possig), 'poisson'))
            pflag = 'poisson win {0} pos'.format(str(w))
            nflag = 'poisson win {0} neg'.format(str(w))
            activity_df[pflag] = possig
            activity_df[nflag] = negsig
        
        if 'quasipoisson' in test:
            tested = True
            ind = test.index('quasipoisson')
            pval = 'quasipoisson pval compare win {0} - {1}'.format(str(w), nWin)
            
            segdisper = activity_df[spave]/activity_df[spvar]
            
            activity_df[pval] = poisson.cdf(activity_df[ave]*dur*nSeg/1000/segdisper, mu=activity_df[spave]*dur*nSeg/1000/segdisper)
            
            pospval = 1 - activity_df[pval]
            
            possig = pospval <= siglvls[ind] #positive significance
            negsig = activity_df[pval] <= siglvls[ind] #negative significance
            
            fr_limit = activity_df[ave]*dur/1000 >= minspike
#             possig[fr_limit] = False 
#             negsig[fr_limit] = False
            
            possig[activity_df[var]==0]=False
            negsig[activity_df[var]==0]=False
            print('Window {0}: Identified significant reposive neurons that {1} increased and {2} decreased firing rates out of {3} with {4} test.'.format(str(w), np.sum(possig), np.sum(negsig),len(possig), 'quasipoisson'))
            pflag = 'quasipoisson win {0} pos'.format(str(w))
            nflag = 'quasipoisson win {0} neg'.format(str(w))
            activity_df[pflag] = possig
            activity_df[nflag] = negsig
        
        if 'nbinom' in test:
            tested = True
            ind = test.index('nbinom')
            pval = 'nbinom pval compare win {0} - {1}'.format(str(w), nWin)
            n = activity_df[spave]**2 * dur*nSeg/1000/(activity_df[spvar]-activity_df[spave])
            p = activity_df[spave]/activity_df[spvar]
            activity_df[pval] = nbinom.cdf(activity_df[ave]*dur*nSeg/1000, n, p)
            
            pospval = 1 - activity_df[pval]
            
            possig = pospval <= siglvls[ind] #positive significance
            negsig = activity_df[pval] <= siglvls[ind] #negative significance
            
            fr_limit = activity_df[ave]*dur/1000 >= minspike
#             possig[fr_limit] = False 
#             negsig[fr_limit] = False
            
            possig[activity_df[var]==0]=False
            negsig[activity_df[var]==0]=False
            print('Window {0}: Identified significant reposive neurons that {1} increased and {2} decreased firing rates out of {3} with {4} test.'.format(str(w), np.sum(possig), np.sum(negsig),len(possig), 'nbinom'))
            pflag = 'nbinom win {0} pos'.format(str(w))
            nflag = 'nbinom win {0} neg'.format(str(w))
            activity_df[pflag] = possig
            activity_df[nflag] = negsig
        
        assert tested, "Testing distribution not specified or unknown testing distribution.\n\
                           Possible tests: poisson, quasipoisson or nbinom"
            
    return activity_df


# if __main__:

#     # code to run the anaylsis
#     stimdir = '/home/feldheimlab/python/Auditory/stimgen/muscimol/'

#     stim_dict = {'seg1':'fullfield.txt',
#     #            'seg2':'randchord.txt',
#                  'seg3':'puretones.txt',
#                  'seg4':'BLN_original.txt',
#                  'seg5':'horizontal_pupcall_1_120rep.txt',
#                  'seg6':'fullfield.txt',
#     #            'seg7':'randchord.txt',
#                  'seg8':'puretones.txt',
#                  'seg9':'BLN_original.txt',
#                  'seg10':'horizontal_pupcall_1_120rep.txt'
#                 }

#     seg = 1 #### CHANGE THIS PARAMTER ONLY
#     noi = 3 #test neuron
#     timewindow = [0,1000] #all time parameters given in ms
#     timeBinSz = 20    


        
#     for key, value in stim_dict.items():
#         print('New trial:', key, value)
#         file = value
#         seg = int(key[-1:])
#         if seg == 0:
#             seg = int(key[-2:])
            
#         stims, num_stim, stim_ind = readStimFile(stimdir, file)

#         if len(num_stim) != len(stim_ind):
#             n_stim = 1
#             for stim in num_stim:
#                 n_stim *= len(stim)
#             print('Number of stimulations: ', n_stim)
#             print(num_stim)
#         else:
#             print('Number of stimulations: ', len(num_stim))
#             print(num_stim)

#         print('\nStimulations order: ', stims.shape)
#         print(stims)

#         print('\nUnique indices: ', stim_ind.shape)
#         print(stim_ind)
        
#         ttls = getTTLseg(seg = seg, ttls=rise, datasets=datasets)
#         pattern, ttlarray = patternGen(asdf, ttls, stims, num_stim, datasets[seg],  window=timewindow, force=True)

#         PatternRaster3d(pattern3d=pattern[0][noi], timerange=timewindow)

#         fr = PatternToCount(pattern=pattern, timeBinSz=timeBinSz, timerange=timewindow, vis=False)

#         nmean = np.squeeze(np.mean(fr[noi]*1000/timeBinSz, axis = 2))
#         nsem = np.squeeze(np.std(fr[noi]*1000/timeBinSz, axis = 2))/np.sqrt(30)

#         fig, axs = plt.subplots(fr.shape[1], fr.shape[2], figsize =(10,2), sharey=True)

#         xs = np.arange(fr.shape[-1])
#         if fr.shape[1]==1:
#             for i in range(fr.shape[2]):
#                 axs[i].errorbar(xs, nmean[i], yerr=[np.zeros_like(nsem[i]), nsem[i]], color='k', linewidth=1)
#                 if i != 0:
#                     axs[i].set_xticklabels([])
#                     axs[i].spines[['left','right', 'top']].set_visible(False)
#                 else:
#                     axs[i].spines[['right', 'top']].set_visible(False)
#             axs[0].set_xticks([xs[0], xs[-1]])
#             axs[0].set_xticklabels(timewindow)
#             plt.subplots_adjust(wspace=0, hspace=0)
#             plt.show()    
#         else:
#             for j in range(fr.shape[1]):
#                 for i in range(fr.shape[2]):
#                     axs[j][i].errorbar(xs, nmean[j][i], yerr=[np.zeros_like(nsem[j][i]), nsem[j][i]], color='k', linewidth=1)
#                     if i != 0:
#                         axs[j][i].set_xticklabels([])
#                         axs[j][i].spines[['left','right', 'top']].set_visible(False)
#                     else:
#                         axs[j][i].spines[['right', 'top']].set_visible(False)
#             axs[fr.shape[1]-1][0].set_xticks([xs[0], xs[-1]])
#             axs[fr.shape[1]-1][0].set_xticklabels(timewindow)
#             plt.subplots_adjust(wspace=0, hspace=0)
#             plt.show()