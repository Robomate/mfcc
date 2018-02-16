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
import matplotlib.pyplot as plt
import matplotlib.image as pltim 
import scipy.io
import pandas as pd
from glob import glob
from PIL import Image
import string
from python_speech_features import mfcc
import scipy.io.wavfile as wav
# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime_1 = timestamp[11:-13]
daytime_2 = timestamp[14:-10]
daytime = daytime_1+"_" +daytime_2
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
path = "/home/root_alien/Desktop/AMI_Lab/" #on alien2
#path = "/home/roboball/Desktop/AMI_Lab/" #on laptop
datapath = 'data/00_VCTK-Corpus_withoutp315'
data_pattern = '**/*.wav'
lab_pattern = '**/p*.txt'
#~ datapath = 'data/eri_german_00_without_badfiles/'
#~ data_pattern = '**/*.wav'
#~ lab_pattern = '**/A*.txt'

subdir = '04_nn_training_audio/'
filepath = path + subdir + datapath #on laptop

# define storage path for model:
model_path = path + subdir + "pretrained_models/"

FIRST_INDEX = ord('a') - 1  # 0 is reserved to space
print(FIRST_INDEX)

# create label dictionary
char_list_down = list(' ')
char_list2 = list(string.ascii_lowercase)
char_list_down = char_list_down + char_list2  + list('_')
print(char_list_down)

# create label dictionary (Capital Letters)
char_list_up = list(' ')
char_list2x = list(string.ascii_uppercase)
char_list_up = char_list_up + char_list2x + list('_')
print(char_list_up)
# create tuple for char dict
char_tuple = (char_list_down,char_list_up)

########################################################################
# init globals
########################################################################

# init unit testing:
unit_test = False        # boolean, default is False

# model parameter
modtype = "RNN_audio_ctc" 
modlayer = 1
save_thres = 0.5 # save only good models	
randomint = 1 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
train_split = 0.9 # train/test split ratio
num_features = 13
num_hidden = 100 # time steps
num_layers = 1
num_classes = 28 # a-z + ' ' +  blank


epochs = 1
learnrate= 1e-4
batchsize_train = 1
batchsize_test = 1

# init activation function:
actdict = ["tanh", "relu", "selu","sigmoid"]
acttype = actdict[1]

########################################################################
# init functions
########################################################################

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

def create_minibatch_ctc(rand_mb, dict_list):
	'''create Minibatch for data and labels
	
	 so far only batchsize = 1 supported !!!!!!!
	'''
	#print(rand_mb)
	print(rand_mb[0][0])
	num_lab = len(rand_mb[1]) # only for entire batch
	#print(num_lab)
	
	###load first audio file###
	# read in .wav-file
	(fs,audio) = wav.read(rand_mb[0][0])
	#print(fs)
	#print(audio.shape)
	# convert to mfcc
	features = mfcc(audio, samplerate=fs)
	#print(features.shape)
	# transform into 3D array
	mb_features = np.asarray(features[np.newaxis, :])
	# normalize features
	mb_features = (mb_features - np.mean(mb_features)) / np.std(mb_features) 
	#print(mb_features.shape)
	seq_length = [mb_features.shape[1]]
	#print(seq_length)
	###load first label###
	delim = ';' # eri_german_00
	#delim = ' ' # vctk
	lab_array = np.genfromtxt(rand_mb[1][0],dtype='str',delimiter=delim)
	lab_array = lab_array[np.newaxis]
	print(lab_array)
		
	# remove different chars
	label1 = ' '.join(lab_array[0].strip().lower().split(' ')).replace('.', '')\
	           .replace('?', '').replace(',', '').replace("'", '')\
	           .replace('!', '').replace('-', ' ').replace('_', ' ')\
	           .replace('(', ''.replace(')', '').replace('"', '').replace('}', ''))
	label = label1.replace(')', '').replace('"', '')
	lab_char_list = list(label)
	#print(lab_char_list)
	
	# if scalar is in label save and skip
	dict_list_num = list(range(0, 9))
	#print(dict_list_num)
	fail_list = []
	for label in lab_char_list:
		valx = [i for i, x in enumerate(dict_list_num) if x==label]
		if valx != []:
			fail_list.append(valx,rand_mb[1][0])
			print(fail_list)
		
	num_char = len(lab_char_list) # number of chars in label
	# create ctc sparse tuple: label = (inidices, values, d_shape)
	indices = np.zeros([num_char,2], dtype=np.int64)
	indices[:,0] = 0
	indices[:,1] = np.arange(0,num_char)
	# convert chars to skalar values
	values = np.empty([0], dtype=np.int32)
	for label in lab_char_list:
		val = [i for i, x in enumerate(dict_list[0]) if x==label] or\
		      [i for i, x in enumerate(dict_list[1]) if x==label]
		if val != []:
			#print(val[0])
			values = np.append(values,val)
	d_shape = np.array([1, num_char], dtype=np.int32)
	#print('created tuple: ctc sparse label')
	#print(indices)
	#print(values)
	#print(d_shape)
	mb_label = (indices, values, d_shape) # create label tuple
	return mb_features, mb_label, seq_length, lab_array[0]
    
def train_loop(epochs, train_tuple, batchsize,dict_list, unit_test):
	''' train the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()	
	# init cost, accuracy:
	train_cost = 0
	train_ler = 0
	#~ cost_history = np.empty(shape=[0],dtype=float)
	#~ train_acc_history = np.empty(shape=[0],dtype=float)
	#~ crossval_history = np.empty(shape=[0],dtype=float)
	# calculate num of batches
	num_samples = len(train_tuple[0])
	batches = int(num_samples/batchsize)
	print('batches',batches)
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 5
		batchsize = 2
	# epoch loop:
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		train_rand_wav = random_shuffle_list(epoch, train_tuple[0])
		train_rand_lab =random_shuffle_list(epoch, train_tuple[1])
		for pos in range(batches):
			print('pos',pos)
			# iterate over rand list for mb
			rand_mb_wav = train_rand_wav[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb_lab = train_rand_lab[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb = (rand_mb_wav,rand_mb_lab)
			#print(rand_mb_wav)
			#print(rand_mb_lab)
			# load batches
			train_audio_mb, train_lab_mb, train_seq_length, original = create_minibatch_ctc(rand_mb, dict_list)
			# start feeding data into model
			feedtrain = {inputs: train_audio_mb,
                         targets: train_lab_mb,
                         seq_len: train_seq_length }
			optimizer.run(feed_dict = feedtrain)
			# get stats
			batch_cost, _ = sess.run([cost, optimizer], feedtrain)
			train_cost += batch_cost * batchsize
			train_ler += sess.run(ler, feed_dict=feedtrain) * batchsize

			# Decoding
			d = sess.run(decoded[0], feed_dict=feedtrain)
			str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
			
			print('Original: %s' % original)
			print('Decoded: %s' % str_decoded)
			
			
		train_cost /= num_samples
		train_ler /= num_samples
		log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, "
		print(log.format(epoch, epochs, train_cost, train_ler))
			
			
			
			# record histories
			#~ ce_loss = sess.run(cross_entropy,feed_dict=feedtrain)
			#~ cost_history = np.append(cost_history, ce_loss)
			#~ train_acc = sess.run(accuracy,feed_dict=feedtrain)
			#~ train_acc_history = np.append(train_acc_history,train_acc)	
			#~ #check progress: training accuracy
			#~ if  pos%4 == 0:
				#~ # start feeding data into the model:
				#~ feedval = {inputs: train_audio_mb,
                         #~ targets: train_lab_mb,
                         #~ seq_len: train_seq_length }
				#~ train_accuracy = accuracy.eval(feed_dict=feedval)
				#~ crossvalidationloss = sess.run(cross_entropy,feed_dict=feedval)
				#~ crossval_history = np.append(crossval_history,crossvalidationloss)
				#~ # print out info
				#~ t1_2 = datetime.datetime.now()
				#~ print('epoch: '+ str(epoch)+'/'+str(epochs)+
				#~ ' -- training utterance: '+ str(pos * batchsize)+'/'+str(num_samples)+
				#~ " -- loss: " + str(ce_loss)[0:-2])
				#~ print('training accuracy: %.2f'% train_accuracy + 
				#~ " -- training time: " + str(t1_2-t1_1)[0:-7])
	#~ print('=============================================')
	#~ #Total Training Stats:
	#~ total_trainacc = np.mean(train_acc_history, axis=0)
	#~ print("overall training accuracy %.3f"%total_trainacc)       
	#~ t1_3 = datetime.datetime.now()
	#~ train_time = t1_3-t1_1
	#~ print("total training time: " + str(train_time)[0:-7]+'\n')
	#~ # create return list
	#~ train_list = [train_acc_history, cost_history, crossval_history, 
			 #~ train_time, total_trainacc]
	#~ return train_list


########################################################################
# init and define model:
########################################################################

inputs = tf.placeholder(tf.float32, [None, None, num_features])
# SparseTensor required by ctc_loss
targets = tf.sparse_placeholder(tf.int32)
# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

# Defining the cell
# Can be:
#   tf.nn.rnn_cell.RNNCell
#   tf.nn.rnn_cell.GRUCell
cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
# Stacking rnn cells
stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers,
									state_is_tuple=True)
# The second output is the last state and we will no use that
outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

shape = tf.shape(inputs)
batch_s, max_time_steps = shape[0], shape[1]

# Reshaping to apply the same weights over the timesteps
outputs = tf.reshape(outputs, [-1, num_hidden])

# Truncated normal with mean 0 and stdev=0.1
# Tip: Try another initialization
# see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
W = tf.Variable(tf.truncated_normal([num_hidden,
									 num_classes],
									stddev=0.1))
# Zero initialization
b = tf.Variable(tf.constant(0., shape=[num_classes]))
# Doing the affine projection
logits = tf.matmul(outputs, W) + b
# Reshaping back to the original shape
logits = tf.reshape(logits, [batch_s, -1, num_classes])
# Time major
logits = tf.transpose(logits, (1, 0, 2))

loss = tf.nn.ctc_loss(targets, logits, seq_len,ignore_longer_outputs_than_inputs=True)
cost = tf.reduce_mean(loss)

# optimizer = tf.train.AdamOptimizer().minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.005, momentum=0.9).minimize(cost)

# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
									  targets))
									  
# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer()) 

# print out model info:
print("**********************************")
print("model: "+str(modlayer)+ " hidden layer "+str(modtype))
print("**********************************")
print("hidden units: "+str(num_hidden)+" each layer")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))



########################################################################
# init main: 
########################################################################	
	
# read in audio data
file_list_wav = find_files(filepath, pattern = data_pattern)
# read in label names
file_list_lab = find_files(filepath, pattern = lab_pattern)	 

# unit-test: assert same num of labels and data
lab_len = len(file_list_lab)
wav_len = len(file_list_wav)
if  lab_len != wav_len :
	msg="Attention!!, Number of labels and data-files does not match!!"
	print(msg)
	print('Difference', abs(lab_len -wav_len ))
	
	#~ # compare files names and print differences
	#~ file_list_wav2 = [s[-12:-4] for s in file_list_wav]
	#~ #print(file_list_wav2)
	#~ file_list_lab2 = [s[-12:-4] for s in file_list_lab ]
	#~ #print(file_list_lab2)
	#~ diff = list(set(file_list_wav2) - set(file_list_lab2))
	#~ for d in diff:
		#~ print(d)
	#~ print()
	#~ print('Assert differences:',len(diff))

	
#~ print(len(file_list_wav))
#~ print(len(file_list_lab))	
#~ print(file_list_wav[100])
#~ print(file_list_lab[100])	
	
# random shuffle data
file_list_wav_rand = random_shuffle_list(1, file_list_wav)
file_list_lab_rand = random_shuffle_list(1, file_list_lab)
file_list = (file_list_wav_rand ,file_list_lab_rand)
#~ print(file_list_wav_rand[100])
#~ print(file_list_lab_rand[100])	
	
# split up training/testing (e.g. 90:10)
train_tuple, test_tuple = split_dataset(train_split, file_list)
#~ print(len(train_tuple[1]))
#~ print(len(test_tuple[0]))
#~ print(train_tuple[0][5])
#~ print(train_tuple[1][5])
#~ print(test_tuple[0][-1])
#~ print(test_tuple[1][-1])

# start training
train_list = train_loop(epochs, train_tuple, batchsize_train,char_tuple, unit_test)








