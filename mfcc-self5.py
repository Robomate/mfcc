#!/usr/bin/env python
#coding: utf-8

# http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/
'''values: -32768 to 32767, 16 bit, max 0dB, SNR 96.33 dB'''
from ctypes import *
from contextlib import contextmanager
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import audioop
import sounddevice as sd
import time

def loadrawdata(filename):
	"""load data from binary .raw format"""
	with open(filename, 'rb') as fid:
		datafile = np.fromfile(fid, dtype=np.int16) #get frames
	return datafile
	
def wavrecord(fs,duration = 3, ch = 1):
	"""wave recorder from mic to numpy arrays"""
	print('start speaking for '+str(duration)+' sec')	
	recording = sd.rec(int(duration * fs), samplerate=fs, channels=ch, blocking=True, dtype='int16')
	sd.wait()
	return recording

def wavplayer(audio,fs):
	"""wave player for numpy arrays"""
	sd.play(audio, fs, blocking=True)
	sd.stop()
	

def plotting(audio, fs, plotnum, subplotnum, plotname, 
			 plotx='time [s]',ploty='amplitude', colors ='b'):
	"""plot audio"""
	plt.figure(plotnum, figsize=(8,8))
	plt.subplots_adjust(wspace=0.5,hspace=0.5)
	plt.subplot(subplotnum)
	plt.plot(range(len(audio)), audio, color = colors)
	plt.axis([0,len(audio),int(np.min(audio))-1,int(np.max(audio))+1])
	if plotnum == 1:
		plt.xticks([w* fs for w in range(int(len(audio)/fs)+1)],
		           [w for w in range(int(len(audio)/fs)+1)])
	plt.title(plotname)
	plt.xlabel(plotx)
	plt.ylabel(ploty)
  
def preemphasis(signal,coeff=0.95):
    """perform pre-emphasis"""    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

def winfft(audio, fs, frame_time=25,step_time=10, window=np.hamming):
	"""apply sliding window, compute FFT"""
	magfft_list = []
	powfft_list = []
	audio_len = len(audio)
	frame_size = int(fs * (frame_time * (1.0/1000)))
	frame_step = int(fs * (step_time* (1.0/1000)))
	
	win = window(frame_size)
	num_fft = 0
	k = 0
	#get fft size
	while frame_size > num_fft:
		num_fft = np.power(2,k) 
		k+=1	
	pos = 0
	#zero pad ftt
	while((pos * frame_step + frame_size) < audio_len):
		pos +=1
	diff = pos * frame_step + frame_size - audio_len
	audio_padded = np.append(audio, np.zeros(diff,np.int16))	
	#compute fft for each chunk
	for posx in range(pos+1):
		audio_chunk = audio_padded[posx * frame_step : 
			posx * frame_step + frame_size].astype(np.float64)
		audio_chunk *= win
		magfft_chunk = np.absolute(np.fft.rfft(audio_chunk,num_fft))
		powfft_chunk = (1.0/num_fft) * np.square(magfft_chunk)	
		magfft_list.append(magfft_chunk)
		powfft_list.append(powfft_chunk)		
	return magfft_list,powfft_list, win, num_fft 

def melbank(fs, numbins=26,lowfreq=0, num_fft=None):
	"""create Mel-filterbank (26 standard, or 26-40 bins)"""
	highfreq = fs/2
	mel_low = 2595 * np.log10(1+lowfreq/700.)
	mel_high  = 2595 * np.log10(1+highfreq/700.)
	mel_peaks = np.linspace(mel_low,mel_high,numbins+2)
	hz_peaks = 700*(10**(mel_peaks/2595.0)-1)
	bin = np.floor((num_fft+1)*hz_peaks/fs)
	#create mel-bank
	melbank = np.zeros([numbins,num_fft//2+1])
	for j in range(0,numbins):
		for i in range(int(bin[j]), int(bin[j+1])):
			melbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
		for i in range(int(bin[j+1]), int(bin[j+2])):
			melbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
	return melbank 
	
def melspec(powfft_list, melbank):
	"""compute 26-40 log-mel spectral + 1 log-energy features"""
	spectrogram = np.zeros([5,5]) #noch falsch
	for powfft in powfft_list:
		#log-melspectrals
		melspec = np.dot(powfft,melbank.T) 
		mfcc = np.where(melspec == 0,np.finfo(float).eps,melspec)
		log_melspec = np.log(melspec)		
		#log-energy
		energy = np.sum(powfft,0) 
		energy = np.where(energy == 0, np.finfo(float).eps,energy)
		log_energy = np.log(energy)					
	return spectrogram
	
def mfcc(melspec_list,numcep=12):
	"""compute 12 MFCC features"""
	for melspec in melspec_list:
		mfcc = dct(melspec, type=2, axis=0, norm='ortho')[:numcep]
	return mfcc

	
#~ def dctopt(cepstra, lift_fac=22):
    #~ """increase magnitude of high frequency DCT coeffs """
    #~ if lift_fac > 0:
        #~ nframes,ncoeff = np.shape(cepstra)
        #~ n = np.arange(ncoeff)
        #~ lift = 1 + (lift_fac/2.)*np.sin(np.pi*n/lift_fac)
        #~ return lift*cepstra
    #~ else:
        #~ # values of lift_fac <= 0, do nothing
        #~ return cepstra  	
	

#~ def delta(feat, N):
    #~ """compute 13 delta + 13 delta_delta features"""
    #~ if N < 1:
        #~ raise ValueError('N must be an integer >= 1')
    #~ NUMFRAMES = len(feat)
    #~ denominator = 2 * sum([i**2 for i in range(1, N+1)])
    #~ delta_feat = numpy.empty_like(feat)
    #~ padded = numpy.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    #~ for t in range(NUMFRAMES):
        #~ delta_feat[t] = numpy.dot(numpy.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    #~ return delta_feat



# init params
frame_time = 25
step_time = 10
num_bins = 26 #Mel bins
low_freq = 0
num_fft = 512

fs = 16000  #Hz
duration = 5  # seconds

#=======================================================================
# read in input
#=======================================================================

# read in audio: .wav files
#~ (fs,audio) = wav.read('artos_ofenrohr_16k.wav')

# read in audio: .raw files	
#filename = 'rolladen_rauf.raw'
filename = 'rolladen_runter.raw'
audio = loadrawdata(filename)

# record audio: extranal mic
#~ audio = wavrecord(fs, duration)

#=======================================================================
# MFCC processing pipeline
#=======================================================================	

# preemphasis
audio_pre = preemphasis(audio)
# windowing, fft
magfft_list,powfft_list, win, num_fft = winfft(audio, fs, frame_time, step_time)
# mel filterbank
melbank = melbank(fs, num_bins, low_freq, num_fft)
# log-melspec + log-energy features
melspec_list = melspec(powfft_list, melbank)
# 12 MFCC features
#~ dct_mfcc = mfcc(melspec_list,numcep=12)
#optimize MFCC features:
 #~ dctopt_mfcc = dctopt(dct_mfcc,lift_fac)
 
# 13 delta + 13 delta_delta features
#~ delta(features, N)

#=======================================================================
# play audio
#=======================================================================

#~ wavplayer(audio,fs)
#~ wavplayer(audio3,fs)

#=======================================================================
#plot files:
#=======================================================================

#~ plotting(audio, fs,1, 211, 'audio .wav file')
#~ plotting(audio_pre, fs,1, 212,'pre-emphasis')
#plotting(win, fs,2,211,'hamming window','samples')
#~ plotting(magfft_list[0], fs,3,211,'fft magnitude','samples')
#~ plotting(powfft_list[0], fs,3,212,'fft power','samples')

plt.show()
plt.clf()
