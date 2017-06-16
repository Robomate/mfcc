#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''=======================================================================
#Purpose: Acoustic Model (Speech Recognition)
#
#Model:   LSTM
#         500 nodes each layer, trained with Adam
#
#Inputs:  Bavarian Speech Corpus (German)
#		  training utterances: 		56127
#		  validation utterances: 	 7012
#         test utterances: 		 	 7023
#         --------------------------------
#		  total					    70162
#
#		  shape of input: 39 MFCCcoeff vector
#
#Output:  135 classes (45 monophones with 3 HMM states each)
#Version: 4/2017 Roboball (MattK.)

#Start tensorboard via bash: 	tensorboard --logdir /logfile/directory
#Open Browser for tensorboard:  localhost:6006
#tensorboard --logdir /home/praktiku/korf/speechdata/tboard_logs/MLP_5layer_2017-05-10_14:04
#======================================================================='''

import numpy as np
import tensorflow as tf
import random
import re
import datetime 
import matplotlib.pyplot as plt
import matplotlib.image as pltim
import os
import scipy.io
import h5py
import pandas

try:
	import cPickle as pickle
except:
   import _pickle as pickle

# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

########################################################################
# define functions:
########################################################################

def loadNamefiles(filenames):
	'''load dataset-names from .txt files:'''
	#load dataset-names from .txt files:
	return np.genfromtxt(filenames, delimiter=" ",dtype='str')

def loadPatData(filename, headerbytes):
	'''load Pattern files:x num of coefficients, 12byte header'''
	with open(filename, 'rb') as fid:
		frames = np.fromfile(fid, dtype=np.int32) #get frames
		#print (frames[0])
		fid.seek(headerbytes, os.SEEK_SET)  # read in without header offset
		datafile = np.fromfile(fid, dtype=np.float32).reshape((frames[0], coeffs)).T 
	return datafile

def createRandomvec(randomint, epochlength):
	'''create random vector'''
	np.random.seed(randomint)
	randomvec = np.arange(epochlength)
	np.random.shuffle(randomvec)
	return randomvec
	
def randomShuffleData(randomint, dataset_name):
	'''based on createRandomvec shuffle data randomly'''
	datasetlength = len(dataset_name)
	#init random list
	dataset_rand = []
	#create random vector
	randomvec = createRandomvec(randomint, datasetlength)
	#fill random list
	for pos in range(datasetlength):
		dataset_rand.append(dataset_name[randomvec[pos]])
	return dataset_rand

def padZeros(datafile):
	'''pad data with zeros or cut data to layer length'''
	data_len = datafile.shape[1]
	if	data_len < nodes:
		# pad data with zeros to layer length
		pad_len = nodes - data_len
		zero_pad = np.zeros([coeffs,pad_len])
		data_padded = np.concatenate((datafile, zero_pad), axis=1)
	elif data_len > nodes:
		# cut data to layer length
		data_padded = datafile[:,0:nodes]		
	else:
		# process data unchanged
		data_padded = datafile
	return data_padded.T

def createMinibatchData(batsize_train,nodes,coeffs):
	'''create Minibatch with input[B_size, Seq_Length, Coeffs]'''
	minibatch_data = np.zeros([batsize_train,nodes,coeffs])
	for batch in range(batsize_train):
		print(batch)
		datafile = loadPatData(datafilename, headerbytes)	
		data_padded = padZeros(datafile)
		minibatch_data[batch,:,:] = data_padded
	return minibatch_data
		
		
def createMinibatchLabel(minibatchsize , classnum, labelbuffer):
	'''create one hot encoding from labels'''
	#init labels with zeros
	minibatchlabel = np.zeros(shape=(minibatchsize,classnum))	
	#one-hot-encoding
	for labelpos in range(minibatchsize):	
		label = int(labelbuffer[labelpos])
		minibatchlabel[labelpos][label-1] = 1.0
	return minibatchlabel

########################################################################
# init parameter
########################################################################

print('=============================================')
print("load filenames")
print('=============================================\n')

# fix initial randomness:
randomint = 1
modtype = "LSTM" 
modlayer = 1	 	

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
path 	 = 'C:/Users/MCU.angelika-HP/Desktop/Korf2017_05/Bachelorarbeit/BA_05/' #on win
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
#path 	 = "/home/korf/Desktop/BA_05/" #on korf@lynx5 (labor)
pathname = "00_data/dataset_filenames/"
logdir = "tboard_logs/"
data_dir = "00_data/pattern_hghnr_39coef/"
label_dir = "00_data/nn_output/"	 
tboard_path = path + logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

# init filenames:
trainset_filename = 'trainingset.txt'
validationset_filename = 'validationset.txt'
testset_filename = 'testset.txt'

#load filenames:
trainset_name = loadNamefiles(path + pathname + trainset_filename)
valset_name = loadNamefiles(path + pathname + validationset_filename)
testset_name = loadNamefiles(path + pathname + testset_filename)
			
# init model parameter:
coeffs = 39  
nodes = 51 							#size of hidden layer
classnum = 135 							#number of output classes
framenum = 1 							#number of frames
inputnum = framenum * coeffs 				#length of input vector
display_step = 100 						#to show progress
bnorm = 'no_bnorm'
lnorm = 'no_lnorm'
	
# init training parameter:
epochs = 1	                            #1 epoch: ca. 1hour10min
learnrate = 1e-4                        #high:1e-4
train_size  = len(trainset_name) 
val_size = len(valset_name)
tolerance = 0.01                        #break condition  
batsize_train = 256													
batches_train = int(train_size / batsize_train) #round down
buffer_train = 10 		#choose number utterances in buffer
train_samples = int(train_size/buffer_train)
train_samples = 10

# init test parameter:
test_size = len(testset_name)	
batsize_test = 100
batches_test = int(test_size / batsize_test)   #round down
buffer_test = 10
test_samples = int(test_size/buffer_test)    #number of test_samples
test_samples = 10


# init emission probs parameter	
batsize_test2=1
buffer_test2=1
test_samples2=10

#random shuffle filenames:
valset_rand = randomShuffleData(randomint, valset_name)
#print(valset_rand)


#init params:		 
headerbytes = 12
datapath  = path + data_dir 
labelpath = path + label_dir
#load data file:
datafilename = datapath + 'is1c0001_001.pat'



#~ datafile = loadPatData(datafilename, headerbytes)	
#~ data_padded = padZeros(datafile)


#input [Batch Size, Sequence Length, Input Dimension]
#(None, 200, 39)
minibatch_data = createMinibatchData(batsize_train,nodes,coeffs)	

print(minibatch_data[0,:,:])	
print(minibatch_data.shape)	

data = tf.placeholder(tf.float32, [None, 20,1])
target = tf.placeholder(tf.float32, [None, 21])
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
