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
from python_speech_features import delta
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

# init unit testing:
unit_test = False        # boolean, default is False

# model parameter
modtype = "RNN_audio_ctc" 
save_thres =  20 # save only good models	0.5
randomint = 1 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
train_split = 0.9 # train/test split ratio
num_features = 39
num_hidden = 128 # time steps
num_layers = 1
num_classes = 28 # a-z + ' ' + blank

epochs = 100 #e.g. 150
learnrate= 5e-3 # use e.g.: 5e-3
momentums = 0.9 # use e.g.: 0.9
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
    
def train_loop(epochs, train_tuple, batchsize,dict_list, unit_test):
	''' train the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()	
	# init cost, accuracy:
	train_cost = 0
	train_ler = 0
	cost_history = np.empty(shape=[0],dtype=float)
	train_acc_history = np.empty(shape=[0],dtype=float)
	crossval_history = np.empty(shape=[0],dtype=float)
	# calculate num of batches
	num_samples = len(train_tuple[0])
	batches = int(num_samples/batchsize)
	print('batches',batches)
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 1
		batchsize = 1
	# epoch loop:
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		train_rand_wav = random_shuffle_list(epoch, train_tuple[0])
		train_rand_lab =random_shuffle_list(epoch, train_tuple[1])
		for pos in range(batches):
			#print('epoch',epoch)
			#print('batches: '+str(pos)+'/'+str(batches))
			# iterate over rand list for mb
			rand_mb_wav = train_rand_wav[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb_lab = train_rand_lab[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb = (rand_mb_wav,rand_mb_lab)
			#print(rand_mb_wav)
			#print(rand_mb_lab)
			# load batches
			train_audio_mb, train_lab_mb, train_seq_length, original_list = create_minibatch_ctc(rand_mb, dict_list)
			# start feeding data into model
			feedtrain = {inputs: train_audio_mb,
                         targets: train_lab_mb,
                         seq_len: train_seq_length }
			optimizer.run(feed_dict = feedtrain)
			# get stats
			batch_cost, _ = sess.run([cost, optimizer], feedtrain)
			train_cost += batch_cost * batchsize
			batch_ler = sess.run(ler, feed_dict=feedtrain) 
			train_ler += batch_ler * batchsize
			# record histories
			cost_history = np.append(cost_history, batch_cost)
			train_acc_history = np.append(train_acc_history,batch_ler)
			
			#check progress: training accuracy
			if  pos%4 == 0:
				######################################
				# test decoding:
				######################################
				feedtest = {inputs: train_audio_mb, 
				           seq_len: train_seq_length}
				# Decoding
				d = sess.run(decoded[0], feed_dict=feedtest)
				print(d[0])
				print(d[1])
				str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
				# Replacing blank label to none
				str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
				# Replacing space label to space
				str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
				pos = 0
				#for original in original_list:
					#pos += 1
					#print('Original ' +str(pos)+':', original)
				print('Original ' +str(pos)+':', original_list[-1])
				print('Decoded: %s' % str_decoded)
				###test validation error###
				feedval = {inputs: train_audio_mb,
                         targets: train_lab_mb,
                         seq_len: train_seq_length }
				train_accuracy = ler.eval(feed_dict=feedval)
				crossvalidationloss = sess.run(loss,feed_dict=feedval)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				# print out info
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(pos * batchsize)+'/'+str(num_samples)+
				" -- loss: " + str(batch_cost)[0:-2])
				print('training error: %.2f'% train_accuracy + 
				" -- training time: " + str(t1_2-t1_1)[0:-7])		
		#~ train_cost /= num_samples
		#~ train_ler /= num_samples
		#~ log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, "
		#~ print(log.format(epoch, epochs, train_cost, train_ler))		
	print('=============================================')
	#Total Training Stats:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training error %.3f"%total_trainacc)       
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')
	# create return list
	train_list = [train_acc_history, cost_history, crossval_history, 
			 train_time, total_trainacc]
	return train_list

def test_loop(epochs,test_tuple, batchsize,dict_list,unit_test):
	'''
	test the neural model
	'''
	print('=============================================')
	print('start '+str(modtype)+' testing')
	print('=============================================')
	t2_1 = datetime.datetime.now()
	# init random numbers
	randomint_test = 1
	random_int = -11
	# init histories
	test_acc_history = np.empty(shape=[0],dtype=float)
	test_filenames_history = np.empty(shape=[0],dtype=str)
	test_softmax_list = []
	label_transcript_list = []
	test_max_transcript_list = []
	# calculate num of batches
	num_samples = len(test_tuple[0])
	batches = int(num_samples/batchsize)
	print('batches',batches)
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 1
		batchsize = 1
	# epoch loop:
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		test_rand_wav = random_shuffle_list(epoch, test_tuple[0])
		test_rand_lab =random_shuffle_list(epoch, test_tuple[1])
		for pos in range(batches):
			#print('epoch',epoch)
			#print('batches: '+str(pos)+'/'+str(batches))
			# iterate over rand list for mb
			rand_mb_wav = test_rand_wav[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb_lab = test_rand_lab[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb = (rand_mb_wav,rand_mb_lab)
			#print(rand_mb_wav)
			#print(rand_mb_lab)
			# load batches
			test_audio_mb, test_lab_mb, test_seq_length, original_list = create_minibatch_ctc(rand_mb, dict_list)
			######################################
			# test decoding:
			######################################
			feedtest = {inputs: test_audio_mb,
			            targets: test_lab_mb, 
			            seq_len: test_seq_length}
			# Decoding
			d = sess.run(decoded[0], feed_dict=feedtest)
			print(d[0])
			print(d[1])
			str_decoded = ''.join([chr(x) for x in np.asarray(d[1]) + FIRST_INDEX])
			# Replacing blank label to none
			str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
			# Replacing space label to space
			str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
			pos = 0
			#~ for original in original_list:
				#~ pos += 1
				#~ print('Original ' +str(pos)+':', original)
			print('Original ' +str(pos)+':', original_list[-1])
			print('Decoded: %s' % str_decoded)
			
			###test validation error###
			test_accuracy = ler.eval(feed_dict=feedtest)
			test_acc_history = np.append(test_acc_history,test_accuracy)
			t2_2 = datetime.datetime.now()
			print('test utterance: '+ str(pos * batchsize)+'/'+str(num_samples))
			print('test error: %.2f'% test_accuracy + 
			" -- test time: " + str(t2_2-t2_1)[0:-7])
					
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(num_samples)+'/'+str(num_samples))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test error %.3f"%total_testacc)       
	t2_3 = datetime.datetime.now()
	test_time = t2_3-t2_1
	print("total test time: " + str(test_time)[0:-7]+'\n')		
	
	# create return list
	test_list  = [test_acc_history, test_filenames_history, 
	              test_softmax_list, randomint_test, total_testacc, 
	              test_time]
	return test_list


########################################################################
# init and define model:
########################################################################
inputs = tf.placeholder(tf.float32, [None, None, num_features])
# SparseTensor required by ctc_loss
targets = tf.sparse_placeholder(tf.int32)
# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])
#######################################
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
########################################
shape = tf.shape(inputs)
batch_s, max_time_steps = shape[0], shape[1]

# Reshaping to apply the same weights over the timesteps
outputs = tf.reshape(outputs, [-1, num_hidden])

# init output weights:
W_out = tf.Variable(tf.truncated_normal([num_hidden,num_classes],stddev=0.1))
b_out = tf.Variable(tf.constant(0., shape=[num_classes]))

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
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)
#~ decoded, log_prob = tf.nn.ctc_beam_search_decoder(
                                                  #~ inputs= logits,
                                                  #~ sequence_length = seq_len,
                                                  #~ beam_width=100,
                                                  #~ top_paths=1,
                                                  #~ merge_repeated=True
                                                  #~ )
# inaccuracy: label error rate
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
print("model: "+str(num_layers)+ " hidden layer "+str(modtype))
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
# start testing: full set
test_list  = test_loop(1,test_tuple, batchsize_test,char_tuple,unit_test)

########################################################################
# plot settings:
########################################################################
# plot model summary:
plot_model_sum = "model_summary: "+str(modtype)+"_"+str(num_layers)+\
                 " hid_layer, "+" time_hid_units: "+str(num_hidden)+\
                 ", vertical_hid_size: "+str(num_layers)+" coeffs: "+\
                 str(num_features)+", \n"+"classes: " +str(num_classes)+\
                 ", epochs: "+str(epochs)+norm+ ", "+\
                 "training error %.3f"%train_list[4]+\
                 " , "+str(timemarker)                         
# define storage path for model:
model_path = path + "04_nn_training_audio/pretrained_models/"
model_name = str(timemarker)+"_"+ modtype + "_"+ str(num_layers)+ "lay_"+\
             str(num_features)+ "fs_"+ str(num_features)+ "coeffs_"+\
             str(num_hidden)+ "hid_units_"+str(num_layers)+"hid_size_"+\
             str(num_classes)+ "class_" +str(epochs)+"eps_"+acttype
              
# init moving average:
wintrain = 300
wintest = 300
wtrain_len = len(train_list[1])
if wtrain_len < 100000:
	wintrain = 5
wtest_len = len(test_list[0])
if wtest_len < 10000:
	wintest = 5
#=======================================================================
# plot training
#=======================================================================

#plot training loss function
fig1 = plt.figure(1, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[1])),train_list[1], color='#1E90FF')
plt.plot(np.convolve(train_list[1], np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(train_list[1]),0,int(200)+1])
#plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batchsize_train))
plt.ylabel('loss')

#plot training error function
plt.subplot(212)
plt.plot(range(len(train_list[0])),train_list[0], color='#20B2AA')
plt.plot(np.convolve(train_list[0], np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(train_list[1]),0,1])
plt.title('Training Error (Character Based)')
plt.xlabel('number of batches, batch size:'+ str(batchsize_train))
plt.ylabel('error percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
#pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

#plot validation loss function
plt.figure(2, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[2])),train_list[2], color='#1E90FF')
plt.plot(np.convolve(train_list[2], np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(train_list[2]),0,int(10)+1])
#plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

#plot test accuracy function
plt.subplot(212)
plt.plot(range(len(test_list[0])),test_list[0], color='m')
plt.plot(np.convolve(test_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(test_list[0]),0,1])
plt.title('Test Error (Character Based)')
plt.xlabel('number of batches, batch size: '+ str(batchsize_test))
plt.ylabel('error percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')

########################################################################
# start export:
########################################################################

# fetch weights from tf.Graph
rnn_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN_cell")
rnn_out = sess.run(rnn_outx)
num_weights = len(rnn_out)
#print weights:
#print('sess.run(tf.get_collection')
print(num_weights)
#print(rnn_outx)
#for weight in rnn_out:
	#print(weight.shape)

# .mat file method:-----------------------------------------------------
# create dict model
model = {"model_name" : model_name}
# modelweights
# RNN cell
pos = 1
pos2 = 1
for weight in rnn_out:
	if pos2%2 == 0:
		model["b"+str(pos)] = weight
		pos += 1
	else:
		model["W"+str(pos)] = weight
	pos2 += 1
# output layer
model["W"+str(pos)] = np.array(sess.run(W_out))
model["b"+str(pos)] = np.array(sess.run(b_out))
model["acttype"] = acttype
# modellayout
model["num_hidden"] = num_hidden
model["num_layers"] = num_layers
model["num_classes"] = num_classes
model["num_features"] = num_features
model["randomint"] = randomint
model["classification"] = classifier 
model["optimizer"] = optimizing
# trainingparams
model["epochs"] = epochs
model["learnrate"] = learnrate
model["batsize_train"] = batchsize_train
model["total_trainacc"] = train_list[4]
# testparams
model["batsize_test"] = batchsize_test
model["randomint_test"] = test_list[3]
model["total_testacc_rvg"] = test_list[4]
# history = [cost_history, train_err_history, test_err_history]
model["cost_history"] = train_list[1]
model["train_err_history"] = train_list[0]
model["test_err_history"] = test_list[0]
# save plot figures
model["fig_trainplot"] = im1
model["fig_testplot"] = im2

# export to .csv file
model_csv = [timemarker,train_list[4],'', modtype,num_layers,
             num_features, '', num_hidden,num_classes,
             classifier,epochs,acttype,learnrate,optimizing,
             norm,norm,batchsize_train,batchsize_test]
df = pd.DataFrame(columns=model_csv)

# save only good models
if train_list[4] < save_thres:
	print('=============================================')
	print("start export")
	print('=============================================')
	print("Model saved to: ")
	print(model_path + model_name)
	scipy.io.savemat(model_path + model_name,model)
	df.to_csv('model_statistics/nn_model_statistics.csv', mode='a')
	#df.to_csv('nn_model_statistics/nn_model_statistics_backup.csv', mode='a')
	print('=============================================')
	print("export finished")
	print('=============================================')
	print(" "+"\n")

# print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: "+ str(modtype))
print("*******************************************")
print("epochs: "+str(epochs))
print("time steps "+str(num_hidden))
print("vertical layers "+str(num_layers))
print("num_features: "+str(num_features))
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(train_list[3])[0:-7])
print("overall training error %.3f"%train_list[4]) 
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_list[5])[0:-7])	
print("overall test error %.3f"%test_list[4]) 
print("*******************************************")
if unit_test is True:
	print('UNIT_TEST finished')
#plot show options:-----------------------------------------------------
#plt.show()
#plt.hold(False)
#plt.close()





