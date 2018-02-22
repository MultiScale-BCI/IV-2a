#!/usr/bin/env python3

'''	Functions used for bandpass filtering and freuquency band generation'''

import numpy as np
from scipy import signal 

__author__ = "Michael Hersche and Tino Rellstab"
__email__ = "herschmi@ethz.ch,tinor@ethz.ch"

def bandpass_filter(signal_in,f_band_nom):
	'''	Filter N channels with fir filter of order 101

	Keyword arguments:
	signal_in -- numpy array of size [NO_channels, NO_samples]
	f_band_nom -- normalized frequency band [freq_start, freq_end]

	Return:	filtered signal 
	'''
	numtabs = 101
	h = signal.firwin(numtabs,f_band_nom,pass_zero=False)
	NO_channels ,NO_samples = signal_in.shape 
	sig_filt = np.zeros((NO_channels ,NO_samples))

	for channel in range(0,NO_channels):
		sig_filt[channel] = signal.convolve(signal_in[channel,:],h,mode='same') # signal has same size as signal_in (centered)

	return sig_filt

def load_bands(bandwidth,f_s):
	'''	Filter N channels with fir filter of order 101

	Keyword arguments:
	bandwith -- numpy array containing bandwiths ex. [2,4,8,16,32]
	f_s -- sampling frequency

	Return:	numpy array of normalized frequency bands
	'''
	if len(bandwidth) == 6: # 1 Hz included 
		f_bands = np.zeros((79,2)).astype(float)
	else:
		f_bands = np.zeros((43,2)).astype(float)

	band_counter = 0
	for bw in bandwidth:
		startfreq = 4
		while (startfreq + bw <= 40): 
			f_bands[band_counter] = [startfreq, startfreq + bw]
			
			if bw ==1:
				startfreq = startfreq +1
			elif bw == 2: 
				startfreq = startfreq +2 
			else : 
				startfreq = startfreq +4

			band_counter += 1 

	# convert array to normalized frequency 
	f_bands_nom = 2*f_bands/f_s
	return f_bands_nom