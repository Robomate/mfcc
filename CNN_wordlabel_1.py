#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#=======================================================================
#Purpose: Classification with a CNN
#
#Model:   4 Layer Convolutional Neural Network
#
#Inputs:  2D UWB radar data (SAR images)
#
#Output:  8 Classes (bed positions)
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

# init unit testing:
unit_test = True      # boolean, default is False

# model parameter
modtype = "CNN" 
modlayer = 4
train_split = 0.9 # train percentage, rest is test
randomint = 2 # fix initial randomness:
norm = 'no_norm'
optimizing = 'adam'
classifier = 'softmax'
save_thres = 0.9 # save only good models	

epochs = 1
learnrate= 1e-4
batchsize_train = 64 * 2
batchsize_test = 10	

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
	print("load filenames")
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

def create_minibatch_wlabel(rand_mb, dict_list):
	'''create Minibatch for data and labels'''
	#print(rand_mb)
	#print(rand_mb[0][0])
	data_max = 500
	classes = len(dict_list)
	num_lab = len(rand_mb[1]) # number of files in batch
	lab_onehot = np.zeros((num_lab, classes))
	print(num_lab)
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
		print(features.shape)
		seq_length = features.shape[0]
		seq_length_list.append(seq_length)
		if seq_length > data_max:
			features = features[0:500,:]
			print("fjfdkfj")
			print(features.shape)
			
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
			print(rand_mb_im)
			print(rand_mb_lab)
			# load batches
			train_im_mb , train_lab_mb, _ = create_minibatch_wlabel(rand_mb, dict_list)
			print(train_im_mb.shape)
			print(train_lab_mb.shape)
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
# init and define model:
########################################################################

# init model:
def weight_variable(shape):
	'''init weights'''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="W")
	
def bias_variable(shape):
	'''init biases'''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name="b")

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
W1 = weight_variable([f1, f1, 1, f1_d])
b1 = bias_variable([f1_d])
W2 = weight_variable([f1, f1, f1_d, f2_d])
b2 = bias_variable([f2_d])
W3 = weight_variable([len_reshape, f3_fc])
b3 = bias_variable([f3_fc])
W4= weight_variable([f3_fc, classes])
b4 = bias_variable([classes])

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
# import data and pre-process data
########################################################################

# read in images
file_list_im = find_files(filepath, pattern='**/*.wav')
# read in label names  
file_list_lab = find_files(filepath, pattern='**/*.txt')

# random shuffle data
file_list_im_rand = random_shuffle_list(1, file_list_im)
file_list_lab_rand = random_shuffle_list(1, file_list_lab)
file_list = (file_list_im_rand ,file_list_lab_rand)
#~ print(file_list_im_rand[100])
#~ print(file_list_lab_rand[100])

# split up training/testing (e.g. 90:10)
train_tuple, test_tuple = split_dataset(train_split, file_list)
#~ print(len(train_tuple[1]))
#~ print(len(test_tuple[0]))
#~ print(train_tuple[0][5])
#~ print(train_tuple[1][5])
#~ print(test_tuple[0][-1])
#~ print(test_tuple[1][-1])

########################################################################
#start main: 
########################################################################

# start training
train_list = train_loop(epochs, train_tuple, batchsize_train,word_list, unit_test)
#~ # start testing
test_list = test_loop(1, test_tuple, batchsize_test,word_list, unit_test)

########################################################################
#plot settings:
########################################################################

# plot model summary:
plot_model_sum = "model_summary: "+str(modtype)+"_"+str(modlayer)+" hid_layer, "+\
                          ", classes: " +str(classes)+", epochs: "+str(epochs)+"\n"+\
                          norm+ ", "+"training accuracy %.3f"%train_list[4]+\
                          " , test accuracy %.3f"%test_list[1]+" , "+\
                          str(timemarker)
                          
model_name = str(timemarker)+"_"+modtype + "_"+ str(modlayer)+ "lay_"+\
             str(classes)+ "class_" +str(epochs)+"eps_"+\
             acttype+"_"+str(test_list[1])[0:-9]+"testacc"
              
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
#plot training
#=======================================================================

#plot training loss function
fig1 = plt.figure(1, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[1])),train_list[1], color='#1E90FF')
#plt.plot(np.convolve(train_list[1], np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(train_list[1]),0,int(10)+1])
#plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batchsize_train))
plt.ylabel('loss')

#plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_list[0])),train_list[0], color='#20B2AA')
#plt.plot(np.convolve(train_list[0], np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(train_list[1]),0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches, batch size:'+ str(batchsize_train))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

#plot validation loss function
plt.figure(2, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(train_list[2])),train_list[2], color='#1E90FF')
#plt.plot(np.convolve(train_list[2], np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(train_list[2]),0,int(10)+1])
#plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

#plot test accuracy function
plt.subplot(212)
plt.plot(range(len(test_list[0])),test_list[0], color='m')
#plt.plot(np.convolve(test_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(test_list[0]),0,1])
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batchsize_test))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')


########################################################################
#start export:
########################################################################

#.mat file method:
model = {"model_name" : model_name} # create dict model
# modelweights
model["W1"] = np.array(sess.run(W1))
model["W2"] = np.array(sess.run(W2))
model["W3"] = np.array(sess.run(W3))
model["W4"] = np.array(sess.run(W4))
# biases
model["b1"] = np.array(sess.run(b1))
model["b2"] = np.array(sess.run(b2))
model["b3"] = np.array(sess.run(b3))
model["b4"] = np.array(sess.run(b4))
# activation type
model["acttype"] = acttype
#train_list = [train_acc_history, cost_history, crossval_history, train_time, total_trainacc]
#test_list  = [test_acc_history, total_testacc, test_time]
# modellayout
model["layer"] = modlayer
model["classes"] = classes
model["randomint"] = randomint
model["classification"] = classifier 
model["optimizer"] = optimizing
model["norm"] = norm
# trainingparams
model["epochs"] = epochs
model["learnrate"] = learnrate
model["batsize_train"] = batchsize_train
model["total_trainacc"] = train_list[4]
# testparams
model["batsize_test"] = batchsize_test
model["total_testacc"] = test_list[1]
# history = [cost_history, train_acc_history, test_acc_history]
model["cost_history"] = train_list[1]
model["train_acc_history"] = train_list[0]
model["test_acc_history"] = test_list[0]

# save plot figures
model["fig_trainplot"] = im1
model["fig_testplot"] = im2

# save radar specific information
#model["word_dict"] = word_list
#model["radar_arena"] = 

# export to .csv file
model_csv = [timemarker,train_list[4],test_list[1],
             modtype,modlayer,classes,
             classifier,epochs,acttype,learnrate,optimizing,
             norm,batchsize_train,batchsize_test]
df = pd.DataFrame(columns=model_csv)
# save only good models
if train_list[4] > save_thres:
	print('=============================================')
	print("start export")
	print('=============================================')
	scipy.io.savemat(model_path + model_name,model)
	df.to_csv('model_statistics/nn_model_statistics.csv', mode='a')
	print("Model saved to: ")
	print(model_path + model_name)
	print('=============================================')
	print("export finished")
	print('=============================================')
	print(" "+"\n")
	
# print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: "+ str(modtype)+"_"+ str(modlayer)+" layer")
print("*******************************************")
print("activation function: "+str(acttype))
print("optimizer: "+str(optimizing))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(train_list[3])[0:-7])
print("overall training accuracy %.3f"%train_list[4]) 
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_list[2])[0:-7])	
print("overall test accuracy %.3f"%test_list[1]) 
print("*******************************************")

#plot show options:-----------------------------------------------------
#plt.show()
#plt.close()








