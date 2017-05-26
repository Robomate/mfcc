import sounddevice as sd
import numpy as np
import time

def loadRawData(filename):
	"""load data from binary .raw format"""
	with open(filename, 'rb') as fid:
		datafile = np.fromfile(fid, dtype=np.int16) #get frames
	return datafile
	
def exportRawData(filename, data):
	"""load data from binary .raw format"""
	with open(filename, 'wb') as fid:
		fid.writelines(data)
		
	
# read in audio: .raw files	
filename = 'is1a0001_043.raw'
fs = 16000
audio = loadRawData(filename)	

#~ print(audio)	
#~ sd.play(audio, fs, blocking=True)
#~ sd.stop()

print('start recording')
duration = 2 # seconds
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, blocking=True, dtype='int16')
time.sleep(1)
print(recording)
sd.play(recording, fs, blocking=True)

time.sleep(1)
filename_ex = 'sechs.raw'
exportRawData(filename_ex, recording)
print('finish export')

#play audio
time.sleep(1)
print('start replay')
audio2 = loadRawData(filename_ex)
sd.play(audio2, fs, blocking=True)
