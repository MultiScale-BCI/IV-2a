#!/usr/bin/env python3

'''	Functions used for calculating the Riemannian features'''

import numpy as np
from pyriemann.utils import mean,base

from filters import bandpass_filter
from eig import gevd

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

def riemann_features(data,f_bands_nom,time_windows ,avg_mat, learning,settings):
	'''	Calculates features using Riemannian covariance matrices 
	for one frequency band and timewindow

	Keyword arguments:
	data -- numpy array of size [NO_trials,channels,time_samples]
	f_bands_nom -- numpy array [start_freq,end_freq]
	time_tindows -- numpy array [start_time,end_time] 
	avg_mat -- precalculated avgerage covariance matrix
	learning --	if True, calculate avg_mat
				if False, use precalculated avg_mat 
	settings -- int in [2,3,4] stands for ["Riemannian","Euclid","NO Adaptation"]


	Return:	cov_avg 	numpy matrix 	size = 22 x 22
			features 	numpy matrix 	size = NO_trials x (23 x 22/2)
	'''	
	NO_channels = len(data[0,:,0])
	NO_trials = len(data[:,0,0])

	t_start = time_windows[0]
	t_end = time_windows[1]
	NO_tsamples = t_end - t_start

	NO_features = ((NO_channels+1)*NO_channels/2)
	NO_features = int(NO_features)
		
	cov_mat = np.zeros((NO_trials,NO_channels,NO_channels))
	features = np.zeros((NO_trials,NO_features))

	#go through all trials 
	for trial in range(0,NO_trials):	
		
		data_filter = bandpass_filter(data[trial,:,t_start:t_end], f_bands_nom)
		
		cov_mat[trial] = 1/(NO_tsamples-1)*np.dot(data_filter,np.transpose(data_filter)) 

	# calculate average covariance matrix 
	if learning == True:
		if settings == 2:
			cov_avg = mean.mean_covariance(cov_mat, metric = 'riemann')
		elif settings == 3:
			cov_avg = mean.mean_covariance(cov_mat, metric = 'euclid')
		else:
			cov_avg = np.eye(22)
	else : 
		if settings == 2 or settings == 3:
			cov_avg = avg_mat
		else:
			cov_avg = np.eye(22)

	# cacluclate the inverse square root of cov_avg
	if settings == 4:
		features[trial] = half_vectorization(base.logm(cov_mat[trial]))
	else:
		cov_avg_sqrt_inv = base.invsqrtm(cov_avg)
		for trial in range(0,NO_trials):
			features[trial] = half_vectorization(base.logm(np.dot(np.dot(cov_avg_sqrt_inv,cov_mat[trial]),cov_avg_sqrt_inv))) #cov_mat[trial])#

	return cov_avg, features 


def riemann_multiband(data,f_bands_nom,time_windows ,avg_mat, learning,settings):
	'''	Calculates features using Riemannian covariance matrices 
	for multiple frequency bands and timewindows

	Keyword arguments:
	data -- numpy array of size [NO_trials,channels,time_samples]
	f_bands_nom -- numpy array [[start_freq1,end_freq1],...,[start_freqN,end_freqN]]
	time_tindows -- numpy array [[start_time1,end_time1],...,[start_timeN,end_timeN]] 
	avg_mat -- precalculated avgerage covariance matrix
	learning --	if True, calculate avg_mat
				if False, use precalculated avg_mat 
	settings -- int in [2,3,4] stands for ["Riemannian","Euclid","NO Adaptation"]


	Return:	cov_avg 	numpy matrix 	size = NO_timewins x NO_freqbands x 22 x 22
			features 	numpy matrix 	size = NO_trials x NO_timewins x NO_freqbands x (23 x 22/2)
	'''
	settings = settings+2
	NO_bands = len(f_bands_nom[:,0])
	time_windows = time_windows.reshape((-1,2))
	NO_time_windows = int(time_windows.size/2)
	NO_channels = len(data[0,:,0])
	NO_trials = len(data[:,0,0])
	NO_features_perMat = ((NO_channels+1)*NO_channels/2)
	NO_features_perMat = int(NO_features_perMat)
	NO_features = NO_features_perMat*NO_time_windows*NO_bands
	
	featmat = np.zeros((NO_trials,NO_time_windows,NO_bands,NO_features_perMat))
	cov_avg = np.zeros((NO_time_windows,NO_bands,NO_channels,NO_channels))
	features = np.zeros((NO_trials,NO_features))

	# iterate through all time windows 
	for t_wind in range(0,NO_time_windows):
		# get start and end point of current time window 
		t_start = time_windows[t_wind,0]
		t_end = time_windows[t_wind,1]

		# iterate through all frequency bandwids 
		for subband in range(0,NO_bands): 
			cov_avg[t_wind,subband], featmat[:,t_wind,subband] = riemann_features(data,f_bands_nom[subband],time_windows[t_wind] ,avg_mat[t_wind,subband], learning,settings)

	features = np.reshape(featmat,(NO_trials,NO_features))

	return cov_avg,features


def half_vectorization(mat):
	'''	Calculates half vectorization of a matrix

	Keyword arguments:
	mat -- symetric numpy array of size 22 x 22
	

	Return:	vectorized matrix 
	'''
	_,N = mat.shape 

	NO_elements = ((N+1)*N/2)
	NO_elements = int(NO_elements)
	out_vec = np.zeros(NO_elements)

	# fill diagonal elements with factor one 
	for diag in range(0,N):
		out_vec[diag] = mat[diag,diag]

	sqrt2 = np.sqrt(2)
	idx = N
	for col in range(1,N):
		for row in range(0,col):
			out_vec[idx] = sqrt2*mat[row,col]
			idx +=1

	return out_vec
