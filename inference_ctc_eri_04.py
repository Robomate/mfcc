#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
#=======================================================================
#Purpose: Classification with a RNN-CTC Network
#
#Model:   RNN architectures
#
#Inputs:  MFCCs
#
#Output:  28 Classes (characters)
#Version: 11/2017 Roboball (MattK.)
#Link: Mfcc: https://github.com/jameslyons/python_speech_features
#=======================================================================
'''
import numpy as np
import tensorflow as tf
import random
import os, re
import datetime
import string 
import matplotlib.pyplot as plt
import matplotlib.image as pltim 
import scipy.io
import pandas as pd
from glob import glob
from PIL import Image
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import sounddevice as sd
import soundfile as sf
import queue
# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime_1 = timestamp[11:-13]
daytime_2 = timestamp[14:-10]
daytime = daytime_1+"_" +daytime_2
date = timestamp[0:-16]
timemarker = date+"_" + daytime

#load pretrained model:
no_pretrain = False    #init with True, for pretrained models with False
# at home
path_matfile = 'pretrained_models/2018-02-03_09_44_RNN_audio_ctc_1lay_39fs_39coeffs_128hid_units_1hid_size_28class_50eps_relu.mat'
#path_matfile = 'pretrained_models/'
#path_matfile = 'pretrained_models/'

# init paths:
#path = "/home/root_alien/Desktop/AMI_Lab/" #on alien2
path = "/home/roboball/Desktop/AMI_Lab/" #on laptop
#~ datapath = 'data/00_VCTK-Corpus_withoutp315'
#~ data_pattern = '**/*.wav'
#~ lab_pattern = '**/p*.txt'
#datapath = 'data/eri_german_00_without_badfiles/'
#datapath = 'data/00_eri_without_badfiles_new/'
datapath = 'data/00_eri_without_badfiles_new_test/'
data_pattern = '**/*.wav'
lab_pattern = '**/A*.txt'
# set delimiter
delim = ';' # eri_german_00
#delim = ' ' # vctk

subdir = '04_nn_training_audio/'
filepath = path + subdir + datapath #on laptop

# define storage path for model:
model_path = path + subdir + "pretrained_models/"

FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
#print(FIRST_INDEX)

# create label dictionary
char_list_down = list(' ')
char_list2 = list(string.ascii_lowercase)
char_list_down = char_list_down + char_list2  + list('_')
#print(char_list_down)

# create label dictionary (Capital Letters)
char_list_up = list(' ')
char_list2x = list(string.ascii_uppercase)
char_list_up = char_list_up + char_list2x + list('_')
#print(char_list_up)
# create tuple for char dict
char_tuple = (char_list_down,char_list_up)

########################################################################
# init globals
########################################################################

# init globals for recording
CHANNELS = 1 # mono:1, stereo:2
RATE = 8000 # fs: 8000 samples per sec, or 44100 cd quality
WAVE_OUT_FILENAME = "test_record.wav"
NUM_RECS = 3 # set number of records
# record as signed int16, max amps: –32.768 bis 32.767
# setup fifo buffer, for audio samples
q = queue.Queue()

# init unit testing:
unit_test = True        # boolean, default is False

# model parameter
modtype = "pretrained_RNN_audio_ctc" 
save_thres =  0.5 # save only good models	0.5
randomint = 1 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
train_split = 0.9 # train/test split ratio
num_features = 39
num_hidden = 128 # time steps
num_layers = 1
num_classes = 28 # a-z + ' ' + blank

epochs = 1 #e.g. 150
learnrate= 1e-4 # use e.g.: 5e-3
momentums = 0.9 # use e.g.: 0.9
batchsize_train = 1
batchsize_test = 1

# init activation function:
actdict = ["tanh", "relu", "selu","sigmoid"]
acttype = actdict[1]

########################################################################
# init functions
########################################################################

def audio_callback(indata, frames, time, status):
	"""This is called (from a separate thread) for each audio block."""
	if status:
		print(status, file=sys.stderr)
	q.put(indata.copy())
	#print('indata')
	#print(indata.shape)
    
def wav_record_listen(fs=8000, ch = 1):
	"""wave recorder from mic to numpy arrays with listener"""
	threshold = 0.08 # set threshold for recording
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

def load_mat_model(datapath):
	'''load pretrained model from .mat file:'''
	mat = scipy.io.loadmat(datapath)
	#get sorted list of dict keys:
	#~ print('mat_keys:')
	#~ for key, value in sorted(mat.items()):
		#~ print (key)
	#~ print('===========================================')
	return mat 

def find_files(directory, pattern='**/*.wav'):
	"""Recursively finds all files matching the pattern."""
	file_list = sorted(glob(os.path.join(directory, pattern), recursive=True))
	print('=============================================')
	print("load filenames:", pattern[-4:], "format")
	print('=============================================')
	print(len(file_list),'data_files loaded\n')
	return file_list

def create_random_vec(randomint, datalength):
	'''create random vector'''
	np.random.seed(randomint)
	randomvec = np.arange(datalength)
	np.random.shuffle(randomvec)
	return randomvec

def random_shuffle_list(randomint, dataset_name):
	'''shuffle list for data randomly'''
	datasetlength = len(dataset_name)
	# init random list
	dataset_rand = []
	# create random vector
	randomvec = create_random_vec(randomint, datasetlength)
	# fill random list
	for pos in range(datasetlength):
		dataset_rand.append(dataset_name[randomvec[pos]])
	return dataset_rand

def split_dataset(train_split, file_list):
	''' split data into training and test set '''
	# opt: create random list and split e.g. 90:10 (train/test)
	num_train = int(len(file_list[0]) * train_split)
	num_test = len(file_list[0]) - num_train
	#~ print(num_train)
	#~ print(num_test)
	# generate tuples for training/testing
	train_list1 = file_list[0][0:num_train]
	train_list2 = file_list[1][0:num_train]
	train_tuple = (train_list1,train_list2)
	test_list1 = file_list[0][num_train:num_train + num_test]
	test_list2 = file_list[1][num_train:num_train + num_test]
	test_tuple = (test_list1,test_list2)
	return train_tuple, test_tuple

def get_wav_mfcc(file_name):
	'''create MFCCs + delta + deltadeltas'''
	# read in .wav-file
	(fs,audio) = wav.read(file_name)
	#print(fs)
	#print(audio.shape)
	# https://github.com/jameslyons/python_speech_features
	mfcc_feat = mfcc(audio, samplerate=fs,winlen=0.025,winstep=0.01,numcep=13,
	nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
	d_mfcc_feat = delta(mfcc_feat, 2) #add delta
	dd_mfcc_feat = delta(d_mfcc_feat, 2) #add deltadeltas
	features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
	#print(mfcc_feat.shape)
	#print(d_mfcc_feat.shape)
	#print(dd_mfcc_feat.shape)
	#print(features.shape)
	return features

def create_minibatch_ctc(rand_mb, dict_list):
	'''create Minibatch for data and labels'''
	#print(rand_mb)
	#print(rand_mb[0][0])
	num_lab = len(rand_mb[1]) # number of files in batch
	#print(num_lab)
	# init lists
	mb_features_list = [] # data features
	seq_length_list = [] # data seqlen 
	num_lab_list = [] # labels 
	num_char_list = [] # labels max len
	value_array = np.empty([0], dtype=np.int32) # labels all values
	indices_array = np.empty([0,2], dtype=np.int32) # labels all indices
	label_list = [] # all labels
	#print(indices_array.shape)
	for pos in range(num_lab):
		################################################
		###load audio files and do feature extraction###
		################################################
		features = get_wav_mfcc(rand_mb[0][pos])
		#print(features.shape)
		#print(features.shape)
		seq_length = features.shape[0]
		seq_length_list.append(seq_length)
		# transform into 3D array
		mb_features = np.asarray(features[np.newaxis, :])
		# normalize features
		mb_features = (mb_features - np.mean(mb_features)) / np.std(mb_features) 
		#print(mb_features.shape)
		mb_features_list.append(mb_features)
		###########################################
		### labels: load labels, create values ####
		###########################################
		lab_array = np.genfromtxt(rand_mb[1][pos],dtype='str',delimiter=delim)
		label_list.append(lab_array)
		lab_array = lab_array[np.newaxis]
		#print(lab_array)
		# remove different chars
		label1 = ' '.join(lab_array[0].strip().lower().split(' ')).replace('.', '')\
		           .replace('?', '').replace(',', '').replace("'", '')\
		           .replace('!', '').replace('-', ' ').replace('_', ' ')\
		           .replace('(', ''.replace(')', '').replace('"', '').replace('}', ''))
		label = label1.replace(')', '').replace('"', '')
		lab_char_list = list(label)
		#print(lab_char_list)
		num_char = len(lab_char_list) # number of chars in label
		#print(num_char)
		num_char_list.append(num_char)
		# create ctc sparse tuple: label = (indices, values, d_shape)
		# convert chars to skalar values
		values = np.empty([0], dtype=np.int32)
		for label in lab_char_list:
			val = [i for i, x in enumerate(dict_list[0]) if x==label] or\
			      [i for i, x in enumerate(dict_list[1]) if x==label]
			if val != []:
				#print(val[0])
				values = np.append(values,val)
		#print(values)
		value_array = np.append(value_array,values)
		
	# init 3D array with zeros and fill with data
	seq_max = max(seq_length_list)
	mb_batch_3D = np.zeros([num_lab,seq_max,features.shape[1]])
	#print(mb_batch_3D.shape)
	for pos2 in range(num_lab):
		### data: create 3D array ####
		#print(mb_features_list[pos2].shape)
		mb_batch_3D[pos,0:mb_features_list[pos2].shape[1],:] = mb_features_list[pos2]
		### labels: create indices ####
		chars = num_char_list[pos2]
		indices = np.zeros([chars,2], dtype=np.int64)
		indices[:,0] = pos2
		indices[:,1] = np.arange(0,chars)
		indices_array = np.append(indices_array,indices, axis=0)
	
	### labels: create d_shape ####		
	num_max = max(num_char_list)		
	d_shape = np.array([num_lab, num_max], dtype=np.int32)
	#print('3D data array')
	#print(mb_batch_3D)
	#print('created tuple: ctc sparse label')
	#print(indices_array)
	#print(indices_array.shape)
	#print(value_array)
	#print(d_shape)
	mb_label = (indices_array, value_array, d_shape) # create label tuple
	return mb_batch_3D, mb_label, seq_length_list, label_list
    
########################################################################
# init and define model:
########################################################################

def weight_variable(shape):
	'''init weights'''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="W")
	
def bias_variable(const, shape):
	'''init biases'''
	initial = tf.constant(const, shape=shape)
	return tf.Variable(initial, name="b")

# init output weights:
W_out = weight_variable([num_hidden,num_classes])
b_out = bias_variable( 0., [num_classes])

# init model:
inputs = tf.placeholder(tf.float32, [None, None, num_features])
# SparseTensor required by ctc_loss
targets = tf.sparse_placeholder(tf.int32)
# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

# multi RNN cells
cells_fw = []
for _ in range(num_layers):
  cell_fw_init = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
  cells_fw.append(cell_fw_init)
# stacking rnn cells
stack = tf.contrib.rnn.MultiRNNCell(cells_fw,state_is_tuple=True)
# calculate cell output, states
with tf.variable_scope("RNN_cell"):
	outputs, rnn_states = tf.nn.dynamic_rnn(stack, inputs,seq_len, dtype=tf.float32)

shape = tf.shape(inputs)
batch_s, max_time_steps = shape[0], shape[1]

# Reshaping to apply the same weights over the timesteps
outputs = tf.reshape(outputs, [-1, num_hidden])
# Doing the affine projection
logits = tf.matmul(outputs, W_out) + b_out
# Reshaping back to the original shape
logits = tf.reshape(logits, [batch_s, -1, num_classes])
# Time major
logits = tf.transpose(logits, (1, 0, 2))
# loss function
loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)
# optimizer:
#optimizer = tf.train.AdamOptimizer(learning_rate=learnrate).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=learnrate, momentum=momentums).minimize(cost)
# Decoder: (Greedy or Beam Search)
#decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                                                  inputs= logits,
                                                  sequence_length = seq_len,
                                                  beam_width=100,
                                                  top_paths=1,
                                                  merge_repeated=True
                                                  )
# inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
									  targets))							  
# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer())

# init output weights:
if no_pretrain is False:
	# load weights/biases
	mat = load_mat_model(path_matfile)
	
	print("assign: Rnn weights")
	# fetch weights from tf.Graph
	rnn_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN_cell")
	#for weight in rnn_outx:
		#print(weight)
	#sess.run is needed to really assign values to matrices!!
	ass1 = sess.run(rnn_outx[0].assign(mat[sorted(list(mat.keys()))[0]]))
	ass2 = sess.run(rnn_outx[1].assign(mat[sorted(list(mat.keys()))[7]][0]))
	ass3 = sess.run(rnn_outx[2].assign(mat[sorted(list(mat.keys()))[1]]))
	ass4 = sess.run(rnn_outx[3].assign(mat[sorted(list(mat.keys()))[8]][0]))
	ass5 = sess.run(W_out.assign(mat[sorted(list(mat.keys()))[2]]))
	ass6 = sess.run(b_out.assign(mat[sorted(list(mat.keys()))[9]][0]))
	
#~ print (ass1)
#~ print (ass2)
#~ print (ass3)
#~ print (ass4)
#~ print (ass5)
#~ print (ass6)
	
# print out model info:
print("**********************************")
print("model: "+str(num_layers)+ " hidden layer "+str(modtype))
print("**********************************")
print("hidden units: "+str(num_hidden)+" each layer")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print("**********************************")

########################################################################
# init main: 
########################################################################

pos5 = 2
# listen to port and record audio streams
while(True):
	# get audio
	audio = wav_record_listen(RATE, CHANNELS)
	#print(audio.shape)
	################################################
	###load audio files and do feature extraction###
	################################################
	#print(audio.shape)
	# https://github.com/jameslyons/python_speech_features
	mfcc_feat = mfcc(audio, samplerate=RATE,winlen=0.025,winstep=0.01,numcep=13,
	nfilt=26,nfft=512,lowfreq=0,highfreq=None,preemph=0.97,ceplifter=22,appendEnergy=True)
	d_mfcc_feat = delta(mfcc_feat, 2) #add delta
	dd_mfcc_feat = delta(d_mfcc_feat, 2) #add deltadeltas
	features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
	#print(mfcc_feat.shape)
	#print(d_mfcc_feat.shape)
	#print(dd_mfcc_feat.shape)
	#print(features.shape)
	seq_length = [features.shape[0]]
	#print(seq_length)
	# transform into 3D array
	mb_features = np.asarray(features[np.newaxis, :])
	# normalize features
	mb_features = (mb_features - np.mean(mb_features)) / np.std(mb_features) 
	#print(mb_features.shape)
	# init 3D array with zeros and fill with data
	mb_batch_3D = np.zeros([1,seq_length[0],features.shape[1]])
	#print(mb_batch_3D.shape)
	### data: create 3D array ####
	mb_batch_3D[0,0:mb_features.shape[1],:] = mb_features
	#print('3D data array')
	#print(mb_batch_3D.shape)
	######################################
	# test decoding:
	######################################
	feedinfer = {inputs: mb_batch_3D, seq_len: seq_length}
	# Decoding
	d = sess.run(decoded[0], feed_dict=feedinfer)
	#print(d[0])
	#print(d[1])
	str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
	# Replacing blank label to none
	str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
	# Replacing space label to space
	str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
	pos = 0
	#~ for original in original_list:
		#~ pos += 1
		#~ print('Original ' +str(pos)+':', original)
	print('Decoded: %s' % str_decoded)
	
	# play audio 
	wav_player(audio,RATE)
	# export audio data
	#wav_write(WAVE_OUT_FILENAME, audio, RATE)
	if pos5 > NUM_RECS:
		print("Total number of recordings: ",pos5-1)
		break	
	pos5 +=1
	
		
# plot last wave record
#~ wav_plot(audio)	
#~ plt.show()








