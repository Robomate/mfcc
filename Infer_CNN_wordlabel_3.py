#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#=======================================================================
#Purpose: Classification with a CNN
#
#Model:   4 Layer Convolutional Neural Network
#
#Inputs:  MFCCs (x,13) features
#
#Output:  5800 Classes (bed positions)
#Version: 11/2017 Roboball (MattK.)
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
path_matfile = 'pretrained_models/2018-02-15_13_25_CNN_4lay_5857class_160eps_relu_0.438testacc.mat'

# init paths:
path = "/home/root_alien/Desktop/AMI_Lab/" #on alien2
#path = "/home/roboball/Desktop/AMI_Lab/" #on laptop
#datapath = 'data/01_test_10000/'
#datapath = 'data/01_data_test/'
#datapath = 'data/data_09_01_2018_10000/'
datapath = 'data/00_eri_wordlabels/'
subdir = '04_nn_training_audio/'
filepath = path + subdir + datapath #on laptop

# define storage path for model:
model_path = path + subdir + "pretrained_models/"

########################################################################
# init globals
########################################################################

# init globals for recording
NUM_RECS = 50 # set number of records
CHANNELS = 1 # mono:1, stereo:2
RATE = 8000 # fs: 8000 samples per sec, or 44100 cd quality
WAVE_OUT_FILENAME = "test_record.wav"
# record as signed int16, max amps: –32.768 bis 32.767
# setup fifo buffer, for audio samples
q = queue.Queue()

# init unit testing:
unit_test = False      # boolean, default is False

# model parameter
modtype = "CNN_pretrained" 
modlayer = 4
train_split = 0.9 # train percentage, rest is test
randomint = 2 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
save_thres = 0.75 # save only good models	

epochs = 40 * 4
learnrate= 1e-4
batchsize_train = 64 * 2
batchsize_test = 10	

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


def find_files(directory, pattern='**/*.wav'):
	"""Recursively finds all files matching the pattern."""
	file_list = sorted(glob(os.path.join(directory, pattern), recursive=True))
	print('=============================================')
	print("load filenames")
	print('=============================================')
	print(len(file_list),'data_files loaded\n')
	return file_list

def get_filelist(filepath):
	filelist = []
	for index, filename in enumerate(sorted(os.listdir(filepath))):
		filelist.append(filename)
		#print '{0:02d}. {1}'.format(index + 1, filename)
	print('=============================================')
	print("load filenames")
	print('=============================================')
	print(len(filelist),'data_files loaded\n')
	return filelist

def load_mat_model(datapath):
	'''load pretrained model from .mat file:'''
	mat = scipy.io.loadmat(datapath)
	#get sorted list of dict keys:
	print('mat_keys:')
	for key, value in sorted(mat.items()):
		print (key)
	print('===========================================')
	return mat 

def load_mat_data(files):
	'''load .mat files'''
	label_data = np.empty(shape=[0],dtype=int)
	label_data_extend = np.empty(shape=[0],dtype=int)
	dict_list = []
	num_im_list = [] # length of image data
	mat = scipy.io.loadmat(datapath + files[0])
	label_dict = mat['label_dict'] # bed positions
	for label in label_dict:
		dict_list.append(label.rstrip())
	classes = len(dict_list)
	print(str(classes)+' Label Categories:')
	for entry in dict_list:
		print(entry)
	print()
	image_data  = np.array(mat['image_tensor'])
	num_im = image_data.shape[2]
	num_im_list.append(num_im)
	label_array = np.array(mat['label'])
	for pos in range(1,len(files)):
		mat = scipy.io.loadmat(datapath + files[pos])
		# load images
		images = np.array(mat['image_tensor'])
		num_im = images.shape[2]
		num_im_list.append(num_im)
		#print(images.shape)
		image_data = np.append(image_data, images, axis=2)
		# load labels
		label = np.array(mat['label'])
		label_array = np.append(label_array, label, axis=0)
	# convert labels to skalar values
	for label in label_array:
		val = [i for i, x in enumerate(dict_list) if x==label]
		if val != []:
			#print(val[0])
			label_data = np.append(label_data,val)		
	# create correct number of labels
	for pos in range(len(num_im_list)):
		lab_array = np.ones(num_im_list[pos]) * label_data[pos]
		label_data_extend = np.append(label_data_extend, lab_array)
	# images: swap axis, convert to 4D tensor
	image_data = np.swapaxes(image_data,0,2)
	image_data = image_data[:,:,:,np.newaxis]
	#number of classes
	classes = len(label_dict)
	# one hot encoding and classes
	label_onehot = one_hot_encoding(label_data_extend,label_dict)		
	return image_data, label_onehot, label_dict, classes

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

def create_minibatch_wlabel(rand_mb, dict_list):
	'''create Minibatch for data and labels'''
	#print(rand_mb)
	#print(rand_mb[0][0])
	data_max = 500 # init maximum length
	classes = len(dict_list)
	num_lab = len(rand_mb[1]) # number of files in batch
	lab_onehot = np.zeros((num_lab, classes))
	#print(num_lab)
	# init lists
	mb_features_list = [] # data features
	seq_length_list = [] # data seqlen 
	label_list = [] # all labels
	#print(indices_array.shape)
	for pos in range(num_lab):
		################################################
		###load audio files and do feature extraction###
		# read in .wav-file
		(fs,audio) = wav.read(rand_mb[0][pos])
		#print(fs)
		#print(audio.shape)
		# convert to mfcc
		features = mfcc(audio, samplerate=fs)
		#+print(features.shape)
		seq_length = features.shape[0]
		seq_length_list.append(seq_length)
		if seq_length > data_max:
			features = features[0:500,:]
			#print("too long",seq_length)
			
		# normalize features
		mb_features = (features - np.mean(features)) / np.std(features)
		#print(mb_features.shape)
		mb_features_list.append(mb_features)
		###########################################
		### labels: load labels, create values ####
		lab_array = np.genfromtxt(rand_mb[1][pos],dtype='str',delimiter='')
		label_list.append(lab_array)
		lab_array = lab_array[np.newaxis]
		#print(int(lab_array[0]))
		#print(lab_array)
		# convert skalar values to one-hot encoding
		lab_onehot[pos][int(lab_array[0])] = 1
		#~ print(lab_onehot)
	#print(seq_length_list)
		
	# init 4D array with zeros and fill with data
	seq_max = max(seq_length_list)
	mb_batch_4D = np.zeros([num_lab,data_max,mb_features.shape[1],1])
	#print(mb_batch_4D.shape)
	for pos2 in range(num_lab):
		### data: create 4D array ####
		#print(mb_features_list[pos2].shape[0])
		bt = np.asarray(mb_features_list[pos2][:,:,np.newaxis])
		mb_batch_4D[pos2,0:mb_features_list[pos2].shape[0],:,:] = bt
	
	#print('4D data array')
	#print(mb_batch_4D)
	return mb_batch_4D, lab_onehot, seq_length_list


def train_loop(epochs, train_tuple, batchsize,dict_list, unit_test):
	''' train the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()	
	# init cost, accuracy:
	cost_history = np.empty(shape=[0],dtype=float)
	train_acc_history = np.empty(shape=[0],dtype=float)
	crossval_history = np.empty(shape=[0],dtype=float)
	# calculate num of batches
	num_samples = len(train_tuple[0])
	batches = int(num_samples/batchsize)
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 2
		batchsize = 3
	# epoch loop:
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		train_rand_im = random_shuffle_list(epoch, train_tuple[0])
		train_rand_lab =random_shuffle_list(epoch, train_tuple[1])
		for pos in range(batches):
			# iterate over rand list for mb
			rand_mb_im = train_rand_im[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb_lab = train_rand_lab[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb = (rand_mb_im,rand_mb_lab)
			#print(rand_mb_im)
			#print(rand_mb_lab)
			# load batches
			train_im_mb , train_lab_mb, _ = create_minibatch_wlabel(rand_mb, dict_list)
			#print(train_im_mb.shape)
			#print(train_lab_mb.shape)
			# start feeding data into model
			feedtrain = {x_input: train_im_mb, y_target: train_lab_mb}
			optimizer.run(feed_dict = feedtrain)
			# record histories
			ce_loss = sess.run(cross_entropy,feed_dict=feedtrain)
			cost_history = np.append(cost_history, ce_loss)
			train_acc = sess.run(accuracy,feed_dict=feedtrain)
			train_acc_history = np.append(train_acc_history,train_acc)	
			#check progress: training accuracy
			if  pos%4 == 0:
				# start feeding data into the model:
				feedval = {x_input: train_im_mb, y_target: train_lab_mb}
				train_accuracy = accuracy.eval(feed_dict=feedval)
				crossvalidationloss = sess.run(cross_entropy,feed_dict=feedval)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				# print out info
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(pos * batchsize)+'/'+str(num_samples)+
				" -- loss: " + str(ce_loss)[0:-2])
				print('training accuracy: %.2f'% train_accuracy + 
				" -- training time: " + str(t1_2-t1_1)[0:-7])
	print('=============================================')
	#Total Training Stats:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training accuracy %.3f"%total_trainacc)       
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')
	# create return list
	train_list = [train_acc_history, cost_history, crossval_history, 
			 train_time, total_trainacc]
	return train_list
		
def test_loop(epochs, test_tuple, batchsize,dict_list, unit_test):
	''' inference: test the neural model '''
	print('=============================================')
	print('start '+str(modtype)+' testing')
	print('=============================================')
	t2_1 = datetime.datetime.now()
	# init histories
	test_acc_history = np.empty(shape=[0],dtype=float)
	test_filenames_history = np.empty(shape=[0],dtype=str)
	
	# calculate num of batches
	num_samples = len(test_tuple[0])
	batches = int(num_samples/batchsize)
	
	# opt: unit testing
	if unit_test is True:
		epochs = 1
		batches = 4
		batchsize = 10
	for epoch in range(1,epochs+1):
		# random shuffle each epoch
		test_rand_im = random_shuffle_list(epoch, test_tuple[0])
		test_rand_lab =random_shuffle_list(epoch, test_tuple[1])
		for pos in range(batches):
			# iterate over rand list for mb
			rand_mb_im = test_rand_im[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb_lab = test_rand_lab[pos * batchsize: (pos * batchsize) + batchsize]
			rand_mb = (rand_mb_im,rand_mb_lab)
			# generate batches
			test_im_mb, test_lab_mb, _ = create_minibatch_wlabel(rand_mb,dict_list)
			
			# start feeding data into model
			feedtest = {x_input: test_im_mb, y_target: test_lab_mb}
			predictions = accuracy.eval(feed_dict = feedtest)
			test_acc_history = np.append(test_acc_history,predictions)
			#check progress: test accuracy
			if  pos%40 == 0:
				test_accuracy = accuracy.eval(feed_dict=feedtest)			
				t2_2 = datetime.datetime.now()
				print('test utterance: '+ str(pos*batchsize)+'/'+str(num_samples))
				print('test accuracy: %.2f'% test_accuracy + 
				" -- test time: " + str(t2_2-t2_1)[0:-7])
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(num_samples)+'/'+str(num_samples))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test accuracy %.3f"%total_testacc)       
	t2_3 = datetime.datetime.now()
	test_time = t2_3-t2_1
	print("total test time: " + str(test_time)[0:-7]+'\n')		
	
	# create return list
	test_list  = [test_acc_history, total_testacc, test_time]
	return test_list

########################################################################
# import data and pre-process data
########################################################################

# get .mat-file names
filelist = get_filelist(filepath)
# load data from .mat files
#image_data, label_onehot, label_dict, classes = load_mat_data(filelist)

########################################################################
# init and define model:
########################################################################

# init model:
def weight_variable(shape, no_pretrain):
	'''init weights'''
	if no_pretrain is True:
		shape = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(shape, name="W")
	
def bias_variable(shape, no_pretrain):
	'''init biases'''
	if no_pretrain is True:
		shape = tf.constant(0.1, shape=shape)
	return tf.Variable(shape, name="b")


def matmul(x, W, b, name="fc"):
	'''matrix vector multiplication'''
	with tf.name_scope(name):
	  return tf.add(tf.matmul(x, W), b)
		
def dense(x, W, b, name="dense"):
	'''matrix vector multiplication'''
	with tf.name_scope(name):
		return tf.add(tf.matmul(x, W), b)

def selu(x):
	'''Selu activation function'''
	with ops.name_scope('elu') as scope:
		alpha = 1.6732632423543772848170429916717
		scale = 1.0507009873554804934193349852946
		return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))
		 
def activfunction(x, acttype):
	'''activation functions'''
	if acttype == "tanh":
		activation_array = tf.tanh(x)
	elif acttype == "relu":
		activation_array = tf.nn.relu(x)
	elif acttype == "selu":
		activation_array = selu(x)	
	else:
		activation_array = tf.sigmoid(x)
	return activation_array

def conv2d(x, W):
	'''convolution operation (cross correlation)'''
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	'''max pooling'''
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
 
# read in label dictionary
word_list = []  
word_file = 'wordlist_unique.csv'
word_path = filepath + word_file
df = pd.read_csv(word_path, header= None)
#print(df)
#print(df.shape)
for index, row in df.iterrows():
	#print(row[0])
	word_list.append(row[0])
#print(df.iloc[-1][0])
word_list
#print(word_list[-1])
#print(len(word_list))

 # load weights/biases	
mat = load_mat_model(path_matfile)  
           
# set conv hyper params:
classes = len(word_list)
im_width = 500
im_height = 13
f1 = 3
f1_d = 32
f2_d = 32 * 2
f3_fc = 512 * 2
len_reshape = 32000

# init weights:
if no_pretrain is False:                   
	W1 = weight_variable(mat[sorted(list(mat.keys()))[0]], no_pretrain)
	b1 = bias_variable(mat[sorted(list(mat.keys()))[8]], no_pretrain)
	W2 = weight_variable(mat[sorted(list(mat.keys()))[1]], no_pretrain)
	b2 = bias_variable(mat[sorted(list(mat.keys()))[9]], no_pretrain)
	W3 = weight_variable(mat[sorted(list(mat.keys()))[2]], no_pretrain)
	b3 = bias_variable(mat[sorted(list(mat.keys()))[10]], no_pretrain)
	W4 = weight_variable(mat[sorted(list(mat.keys()))[3]], no_pretrain)
	b4 = bias_variable(mat[sorted(list(mat.keys()))[11]], no_pretrain)
                       
# init weights:                   
#~ W1 = weight_variable([f1, f1, 1, f1_d])
#~ b1 = bias_variable([f1_d])
#~ W2 = weight_variable([f1, f1, f1_d, f2_d])
#~ b2 = bias_variable([f2_d])
#~ W3 = weight_variable([len_reshape, f3_fc])
#~ b3 = bias_variable([f3_fc])
#~ W4= weight_variable([f3_fc, classes])
#~ b4 = bias_variable([classes])

# init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None,im_width,im_height, 1],name="input")
y_target = tf.placeholder(tf.float32, shape=[None, classes],name="labels")

####### init model ########
# 1.conv layer    
h_conv1 = tf.nn.relu(conv2d(x_input, W1) + b1)
h_pool1 = max_pool_2x2(h_conv1)
# 2.conv layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W2) + b2)
h_pool2 = max_pool_2x2(h_conv2)
#print(h_pool2.shape)
# 1. FC Layer
h_flat = tf.reshape(h_pool2, [-1, len_reshape])
#print(h_flat)
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W3) + b3)
# readout Layer
h_out = tf.matmul(h_fc1, W4) + b4
# define classifier, cost function:
softmax = tf.nn.softmax(h_out)

# define loss, optimizer, accuracy:
cross_entropy = tf.reduce_mean(
tf.nn.softmax_cross_entropy_with_logits(logits= h_out,labels = y_target))
optimizer = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_out,1), tf.argmax(y_target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer()) 

# print out model info:
print("**********************************")
print("model: "+str(modlayer)+ " layer "+str(modtype))
print("**********************************")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print()


########################################################################
#start main: 
########################################################################

# listen to port and record audio streams
pos5 = 2
while(True):
	mb_infer = np.zeros([1,500,13,1])
	#print(mb_infer.shape)
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
	features = mfcc_feat
	#d_mfcc_feat = delta(mfcc_feat, 2) #add delta
	#dd_mfcc_feat = delta(d_mfcc_feat, 2) #add deltadeltas
	#features = np.concatenate((mfcc_feat, d_mfcc_feat, dd_mfcc_feat), axis=1)
	#print(mfcc_feat.shape)
	#print(d_mfcc_feat.shape)
	#print(dd_mfcc_feat.shape)
	#print(features.shape)
	if features.shape[0] > 500:
		features = features[0:500,:]
	# transform into 3D array
	mb_features = np.asarray(features[np.newaxis, :,:,np.newaxis])
	# normalize features
	mb_features = (mb_features - np.mean(mb_features)) / np.std(mb_features) 
	#print(mb_features.shape)
	mb_infer[:,0:mb_features.shape[1],:,:] = mb_features
	######################################
	# test decoding:
	######################################
	feedinfer = {x_input: mb_infer}
	# Decoding
	sf_max_class = sess.run(softmax, feed_dict=feedinfer)
	sf_idx = np.argmax(sf_max_class)
	#print(sf_idx)	
	print('Decoded: %s' % word_list[sf_idx])
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




