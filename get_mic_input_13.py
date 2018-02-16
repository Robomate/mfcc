#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Record a few seconds of audio and save to a WAVE file.
check for streams
https://pypi.python.org/pypi/sounddevice
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import matplotlib.pyplot as plt

# init globals
CHANNELS = 1 # mono:1, stereo:2
RATE = 8000 # fs: 8000 samples per sec, or 44100 cd quality
WAVE_OUT_FILENAME = "test_record.wav"
NUM_RECS = 3 # set number of records
# record as signed int16, max amps: –32.768 bis 32.767
# use to set default values
#sd.default.samplerate = RATE
#sd.default.device = 'digital output'
#sd.default.channels = 1
# setup fifo buffer, for audio samples
q = queue.Queue()

# init functions
def audio_callback(indata, frames, time, status):
	"""This is called (from a separate thread) for each audio block."""
	if status:
		print(status, file=sys.stderr)
	q.put(indata.copy())
	#print('indata')
	#print(indata.shape)
    
def wav_record_listen(fs=8000, ch = 1):
	"""wave recorder from mic to numpy arrays with listener"""
	threshold = 0.08 # e.g.: 0.08 set threshold for recording
	thres2 = 0.014
	rec = 0 # init only
	rec0 = np.empty(shape=[0],dtype=float) # init rec
	# opening input stream
	stream = sd.InputStream(device=None, channels=ch,
	                        samplerate=RATE, callback=audio_callback)
	with stream:
		print('Honina: Hey whats up? I am listing.')
		while (True):
			if len(rec0) > 20000:
				rec0 = rec0[-5000:]
			rec= q.get()
			rec0 = np.append(rec0,rec)
			thres_max = np.mean(abs(rec))
			#print('amplitude: ', thres_max, 'rec threshold',threshold)
			if thres_max > threshold:
				print('Honina: Start speaking please.')
				while (True):
					data = q.get()
					rec = np.append(rec,data)
					thres_min = np.mean(abs(rec[-5000:]))
					#print(thres_min)
					if thres_min < thres2:
						break
				break
		print("Honina: Done, samples recorded:",rec0.shape[0])
	return np.append(rec0[-5000:], rec)

def wav_player(audio,fs):
	"""wave player for numpy arrays"""
	sd.play(audio, fs, blocking=True)
	sd.stop()

def wav_read(filename):
	"""read wav, raw.. files"""
	data, samplerate = sf.read(filename)
	return data, samplerate

def wav_write(filename, audio, fs):
	"""write wav files"""
	sf.write(filename, audio, fs)

def wav_plot(audio):
	"""plot wav files"""
	plt.figure()
	plt.plot(audio)
	plt.yticks([-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1])
	
def main():
	pos = 2
	# listen to port and record audio streams
	while(True):
		audio = wav_record_listen(RATE, CHANNELS)
		# play audio 
		wav_player(audio,RATE)
		# export audio data
		wav_write(WAVE_OUT_FILENAME, audio, RATE)
		if pos > NUM_RECS:
			print("Total number of recordings: ",pos-1)
			break	
		pos +=1
	# plot wave record
	wav_plot(audio)
	
	# load files from disk
	#~ audio2, samplerate = wav_read(WAVE_OUT_FILENAME)
	#~ print(audio2)
	#~ print(audio2.shape)
	#~ wav_player(audio2,RATE)
	#~ wav_plot(audio2)
		
if __name__ == "__main__":
	pass
	main()
	plt.show()
	
	
	
	
	
