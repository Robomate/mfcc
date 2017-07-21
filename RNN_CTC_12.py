#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''=======================================================================
#Purpose: Acoustic Model (Speech Recognition)
#
#Model:   'BASIC_RNN','RNN','BASIC_LSTM','LSTM', 'GRU'
#         e.g. 500 time_steps each layer, trained with Adam
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
import pandas as pd

try:
	import cPickle as pickle
except:
   import _pickle as pickle

# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

########################################################################
# define functions:
########################################################################

def load_name_files(filenames):
	'''load dataset-names from .txt files:'''
	return np.genfromtxt(filenames, delimiter=" ",dtype='str')

def load_pat_data(filename, headerbytes):
	'''load Pattern files:x num of coefficients, 12byte header'''
	with open(filename, 'rb') as fid:
		frames = np.fromfile(fid, dtype=np.int32) # get frames
		# print (frames[0])
		fid.seek(headerbytes, os.SEEK_SET)  # read in without header offset
		datafile = np.fromfile(fid, dtype=np.float32).reshape((frames[0], coeffs)).T 
	return datafile

def create_random_vec(randomint, epochlength):
	'''create random vector'''
	np.random.seed(randomint)
	randomvec = np.arange(epochlength)
	np.random.shuffle(randomvec)
	return randomvec
	
def random_shuffle_data(randomint, dataset_name):
	'''based on createRandomvec shuffle data randomly'''
	datasetlength = len(dataset_name)
	# init random list
	dataset_rand = []
	# create random vector
	randomvec = create_random_vec(randomint, datasetlength)
	# fill random list
	for pos in range(datasetlength):
		dataset_rand.append(dataset_name[randomvec[pos]])
	return dataset_rand

def pad_zeros(data_file, label_file):
	'''pad data and labels with zeros or cut data to layer length'''
	data_len = data_file.shape[1]
	label_len = label_file.shape[0]
	assert data_len == label_len, "Error: Data and Label length differ."	
	if	data_len < time_steps:
		# zero pad data
		pad_len = time_steps - data_len
		data_zeros = np.zeros([coeffs,pad_len])
		data_padded = np.concatenate((data_file, data_zeros), axis=1)
		# zero pad labels
		label_zeros = 100 * np.ones([pad_len])
		label_padded = np.concatenate((label_file, label_zeros), axis=0)
	elif data_len > time_steps:
		# cut data, labels to layer length
		data_padded = data_file[:,0:time_steps]
		label_padded = label_file[0:time_steps]	
	else:
		# process data, labels unchanged
		data_padded = data_file
		label_padded = label_file
	return data_padded.T, label_padded

def create_minibatch_ctc(minibatchsize,time_steps,coeffs,filenames):
	'''create Minibatch for data and labels'''
	minibatch_data = np.zeros([minibatchsize,time_steps,coeffs])

	# init ctc sparse labels
	indices = np.empty([0,2], dtype=np.int64)
	values = np.empty([0], dtype=np.int32)
	d_shape = np.array([minibatchsize, 0], dtype=np.int32)
	# init seq_len
	seq_length = np.zeros([minibatchsize], dtype=np.int64)
	lab_len = 0 
	
	for batch in range(minibatchsize):
		# load data
		f_pat = filenames[batch][:-4]+".pat"
		data_file = load_pat_data(datapath + f_pat, headerbytes)
		# load labels
		label_txt = filenames[batch][:-4]+".txt"
		label_input = labelpath + label_txt
		label_file = np.loadtxt(label_input)
		# zero padding
		data_padded, label_padded = pad_zeros(data_file, label_file)
		minibatch_data[batch,:,:] = data_padded		
		# time_steps vs. length labels
		if len(label_file) > time_steps:
			indices_part = np.zeros([time_steps,2], dtype=np.int64)
			indices_part[:,0] = batch
			indices_part[:,1] = np.arange(0,time_steps)
			values_part = label_file[0:time_steps]
			d_shape[1] = time_steps
			# fill seq_len
			seq_length[batch] = time_steps
		else:
			indices_part = np.zeros([len(label_file),2], dtype=np.int64)
			indices_part[:,0] = batch
			indices_part[:,1] = np.arange(0,len(label_file))
			values_part = label_file
			if d_shape[1] < len(label_file):
				d_shape[1] = len(label_file)
			# fill seq_len
			seq_length[batch] = len(label_file)
		# fill ctc sparse labels	
		indices = np.concatenate((indices, indices_part), axis=0)
		values = np.concatenate((values, values_part), axis=0).astype(np.int32)	
	#create tuple for sparse label tensor
	if d_shape[1] == 0:
		d_shape[1] = time_steps	
	minibatch_label = (indices, values, d_shape)
	return minibatch_data, minibatch_label, seq_length

def phonemeDict():
	'''Dictionary for classes'''
	phoneme_dict = ["@","a","a:","aI","an","ar","aU","b","C","d","e:","E",
	"E:","f","g","h","i:","I","j","k","l","m","n","N","o:","O","Oe","On",
	"OY","p","r","s","S","sil","sp","t","u","u:","U","v","x","y:","Y","z","Z"]
	#create label translation
	phoneme_lab = np.arange(classnum).astype('str')
	pos=0
	for phoneme in phoneme_dict:
		phoneme_lab[pos*3:pos*3+3] = phoneme
		pos += 1
	return phoneme_lab
	
def phonemeTranslator(labels,phoneme_labels):
	'''translate labels into phonemes'''
	label_transcript = np.copy(labels).astype(str)	
	for pos in range(len(labels)):
		phon = phoneme_labels[int(labels[pos])-1]
		label_transcript[pos] = phon	
	#print("labels utterance: "+str(labels))
	#print(label_transcript)	
	return label_transcript

def cell_type(ctype):
	'''set RNN cell type booleans'''
	if ctype == "RNN":
		cell_boolean = [True,False,False,False]			
	elif ctype == "BASIC_LSTM":
		cell_boolean = [False,True,False,False]	
	elif ctype == "LSTM":
		cell_boolean = [False,False,True,False]		
	elif ctype == "GRU":
		cell_boolean = [False,False,False,True]		
	else:
		print("ERROR: wrong cell type!! instead choose:'BASIC_RNN','RNN','BASIC_LSTM','LSTM', 'GRU' ")
		cell_boolean = [False,False,False,False]		
	return cell_boolean

def act_function(acttype):
	'''activation functions'''
	if acttype == 'tanh':
		activation_array = tf.nn.tanh
	elif acttype == 'relu':
		activation_array = tf.nn.relu
	else:
		activation_array = tf.nn.sigmoid
	return activation_array

	
########################################################################
# init parameter
########################################################################

print('=============================================')
print("load filenames")
print('=============================================\n')

#choose cell: ['RNN', 'BASIC_LSTM','LSTM', 'GRU']
cell_number = 2

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
#path 	 = 'C:/Users/MCU.angelika-HP/Desktop/Korf2017_05/Bachelorarbeit/BA_05/' #on win
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
#path 	 = "/home/praktiku/Videos/BA_05/" #on praktiku@dell8 (labor)
path 	 = "/home/korf/Desktop/BA_05/" #on korf@alien2, korf@lynx5 (labor)
pathname = "00_data/dataset_filenames/"

data_dir = "00_data/pattern_hghnr_39coef/"
#data_dir = "00_data/pattern_dft_16k/"
#data_dir = "00_data/pattern_dft_16k_norm/"

label_dir = "00_data/nn_output/"
#label_dir = "00_data/nn_output_tri/"

# init filenames:
trainset_filename = 'trainingset.txt'
validationset_filename = 'validationset.txt'
testset_filename = 'testset.txt'

#load filenames:
trainset_name = load_name_files(path + pathname + trainset_filename)
valset_name = load_name_files(path + pathname + validationset_filename)
testset_name = load_name_files(path + pathname + testset_filename)
			
# init model parameter:
cell_name = ['RNN','BASIC_LSTM','LSTM', 'GRU']
modtype = cell_name[cell_number]    # options: 'RNN', 'BASIC_LSTM'..
modlayer = 1                       # layer deepness of the network
coeffs = 39                         # MFCC: 39, DFT:257
time_steps = 500                   # unit over time
hidden_dim = 100     				# size of hidden layer
classnum = 135 						# 3state mono: 135, triphone: 4643

#noch falsch???????????!!!!!
classnum_ctc = classnum + 1 + 1

framenum = 1 					    # number of input frames
inputnum = framenum * coeffs 		# length of input vector
randomint = 1
display_step = 100 					# to show progress
bnorm = 'no_bnorm'
lnorm = 'no_lnorm'
optimizing = 'adam'
classifier = 'softmax'
acttype = 'tanh'
momentum = 0.99
# customize cell:
#act_func = act_function(acttype)    # init act function
cell_boolean = cell_type(modtype)   # init cell type

# init training parameter:
epochs = 20                       
learnrate = 1e-4                        #high:1e-4
train_size  = len(trainset_name) 
val_size = len(valset_name)
tolerance = 0.01                        #break condition  
batsize_train = 256												
training_batches = int(train_size/batsize_train)

# init test parameter:
test_size = len(testset_name)	
batsize_test = batsize_train   #noch falsch!!!!
test_batches = int(test_size/batsize_test)    #number of test_batches

# init emission probs parameter	
batsize_test2 = 1
test_batches2 = 10

#~ #random shuffle filenames:
#~ valset_rand = random_shuffle_data(randomint, valset_name)
#~ #print(valset_rand)

# init params:		 
headerbytes = 12
datapath  = path + data_dir 
labelpath = path + label_dir

# perform unit testing:
epochs = 1 
training_batches = 100  #number of training batches
test_batches = 10

# init tboard
logdir = "tboard_logs/"	 
tboard_path = path + logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

########################################################################
# init and define model:
########################################################################

# init model functions:
def weight_variable(shape):
	'''init weights'''
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name="W")
	
def bias_variable(shape):
	'''init biases'''
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name="b")

# weights: output layer
with tf.name_scope(name="fc"):
	W_out = weight_variable([hidden_dim, classnum_ctc])
	b_out = bias_variable([classnum_ctc])
	tf.summary.histogram("weights", W_out)
	tf.summary.histogram("biases", b_out)

# init placeholder:
x_input  = tf.placeholder(tf.float32,[None, None, coeffs], name='inputs') 
y_target = tf.sparse_placeholder(tf.int32)
seq_len = tf.placeholder(tf.int32, [None]) 

# init cell variant: ['BASIC_RNN','RNN','BASIC_LSTM','LSTM', 'GRU']
if cell_boolean[0]:
	cell = tf.contrib.rnn.BasicRNNCell(hidden_dim)
elif cell_boolean[1]:
	cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, forget_bias=0.0, state_is_tuple=True)
elif cell_boolean[2]:
	cell = tf.contrib.rnn.LSTMCell(hidden_dim,state_is_tuple=True)
elif cell_boolean[3]:
	cell = tf.contrib.rnn.GRUCell(hidden_dim)
else:
    print("ERROR: wrong cell type!!")
#print(cell)

# stacking rnn cells
stack = tf.contrib.rnn.MultiRNNCell([cell] * modlayer,state_is_tuple=True)
# calculate cell output, states
with tf.variable_scope("RNN_cell"):
	rnn_outputs, rnn_states = tf.nn.dynamic_rnn(stack, x_input,seq_len, dtype=tf.float32)

# reshape rnn_outputs
rnn_outputs_re = tf.reshape(rnn_outputs, [-1, hidden_dim])
# linear output layer
output = tf.matmul(rnn_outputs_re, W_out) + b_out
# reshape back to the original shape
logits_x = tf.reshape(output, [batsize_train, -1, classnum_ctc])
# time major
logits = tf.transpose(logits_x, (1, 0, 2))

# define ctc loss, optimizer:
with tf.name_scope("ctc_loss"):
	loss_ctc = tf.nn.ctc_loss(y_target, logits, seq_len)
	cost = tf.reduce_mean(loss_ctc)
with tf.name_scope("train"):
	optimizer = tf.train.MomentumOptimizer(learnrate, momentum).minimize(cost)

# define ctc decoder:
# option 1:
decoded, log_prob = tf.nn.ctc_greedy_decoder(logits,seq_len,merge_repeated=True)
# option 2:
decoded_1, log_prob_1 = tf.nn.ctc_beam_search_decoder(logits,seq_len,beam_width=100,top_paths=3, merge_repeated=True)

#~ # define label_error_rate:
with tf.name_scope("label_error_rate"):
	ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),y_target))

# merge all summaries for tensorboard:	
summ = tf.summary.merge_all()

# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer()) 
# init tensorboard
#writer = tf.summary.FileWriter(tboard_path + tboard_name)
#writer.add_graph(sess.graph)

# print out model info:
print("**********************************")
print("model: "+str(1)+" hidden layer "+str(modtype))
print("**********************************")
print("epochs: "+str(epochs))
print("hidden units: "+str(time_steps)+" each layer")
print("hidden size: "+str(hidden_dim))
print("optimizer: Adam")
print("----------------------------------")
print("data name: RVG new German speech corpus")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size)+"\n")

########################################################################
# training loop:
########################################################################

def training(epochs,training_batches):
	'''train the neural model'''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()
	
	# init cost, accuracy:
	crossval_history = np.empty(shape=[0],dtype=float)
	cost_history = np.empty(shape=[0],dtype=float)
	train_acc_history = np.empty(shape=[0],dtype=float)

	# epoch loop:
	for epoch in range(1,epochs+1):
		
		# random shuffle filenames for each epoch:
		randomint_train = epoch
		trainset_rand = random_shuffle_data(randomint_train, trainset_name)
		
		# training loop: 
		for minibatch in range(int(training_batches)):
			# grab linear utterances from random trainset:
			trainset_buffer = trainset_rand[minibatch * batsize_train:(minibatch * batsize_train) + batsize_train]
			# minibatch_data [Batch Size, Sequence Length, Input Dimension]
			minibatch_train_data, minibatch_train_label, seq_length = create_minibatch_ctc(batsize_train,time_steps,coeffs,trainset_buffer)
		
			# start feeding data into the model:
			feedtrain_ctc = {x_input: minibatch_train_data,y_target:minibatch_train_label, seq_len: seq_length}
			#~ #batch_cost, _ = sess.run([cost, optimizer], feedtrain_ctc)
			optimizer.run(feed_dict = feedtrain_ctc)
			
			#igorq
			#~ batch_cost, _ = sess.run([cost, optimizer], feedtrain_ctc)
			#~ print(batch_cost)
			#train_cost += batch_cost*batch_size
			#train_ler += sess.run(ler, feed_dict=feedtrain_ctc)*batch_size
			
		
			# log history for tensorboard
			if minibatch % 5 == 0:
				[train_accuracy, s] = sess.run([ler, summ], feed_dict=feedtrain_ctc)
				# writer.add_summary(s, batch)
				
			# get cost_history, accuracy history data:
			cost_history = np.append(cost_history,sess.run(cost,feed_dict=feedtrain_ctc))
			train_acc_history = np.append(train_acc_history,sess.run(ler,feed_dict=feedtrain_ctc))
			
			# check progress: training accuracy
			if  minibatch%10 == 0:
				train_accuracy = ler.eval(feed_dict=feedtrain_ctc)
				crossvalidationloss = sess.run(cost,feed_dict=feedtrain_ctc)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(minibatch * batsize_train)+'/'+str(train_size)+
				" -- cross validation loss: " + str(crossvalidationloss)[0:-2])
				print('training error rate: %.2f'% train_accuracy + 
				" -- training time: " + str(t1_2-t1_1)[0:-7])
				
			#stopping condition:
			#if abs(crossval_history[-1] - crossval_history[-2]) < tolerance:
				#break
		print('=============================================')	
	print('=============================================')
	# total training statistics:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training error rate %.3f"%total_trainacc)       
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')	
	
	return train_acc_history, cost_history, crossval_history, train_time, total_trainacc

########################################################################
# start testing:
########################################################################

def testing(test_batches=10,batsize_test=1, phase_bool=False, randomint_test=1):
	'''test the neural model'''
	print('=============================================')
	print('start '+str(modtype)+' testing')
	print('=============================================')
	t2_1 = datetime.datetime.now()
	
	#init histories
	test_acc_history = np.empty(shape=[0],dtype=float)
	test_filenames_history = np.empty(shape=[0],dtype=str)
	test_softmax_list = []
	label_transcript_list = []
	test_max_transcript_list = []
	
	#get label-dictionary:
	phoneme_labels = phonemeDict()
	if test_batches < 20:
		print("phoneme_labels: ")
		print(phoneme_labels)
		print(" ")
		
	#random shuffle filenames:	
	testset_rand = random_shuffle_data(randomint_test, testset_name)
	# init histories
	test_softmax_utterance = np.empty(shape=[0,classnum],dtype=float)

	#test loop:
	for minibatch in range(int(test_batches)):
		# grab linear utterances from random trainset:
		testset_buffer = testset_rand[minibatch * batsize_test:(minibatch * batsize_test) + batsize_test]
		# minibatch_data [Batch Size, Sequence Length, Input Dimension]
		minibatch_test_data, minibatch_test_label, seq_length = create_minibatch_ctc(batsize_test,time_steps,coeffs,testset_buffer)
	
		# start feeding data into the model:
		feedtest_ctc = {x_input: minibatch_test_data,y_target:minibatch_test_label, seq_len: seq_length}
		
		
		#~ print('testset_buffer')
		#~ print(testset_buffer)
		
		#~ print('minibatch_test_data')
		#~ #print(minibatch_test_data)
		#~ print(minibatch_test_data.shape)
		
		#~ print('minibatch_test_label')
		#~ print(len(minibatch_test_label))
		#~ print(minibatch_test_label)	
		
		
		#predictions funktioniert noch nicht!!!!!!!
		
		
		predictions = ler.eval(feed_dict = feedtest_ctc)
		test_acc_history = np.append(test_acc_history,predictions)
			
		#~ # get label to phoneme transcription:
		#~ #label_transcript = phonemeTranslator(minibatch_test_label, phoneme_labels)
	
		#~ #check progress: test accuracy
		#~ if  minibatch%10 == 0:
			#~ test_accuracy = accuracy.eval(feed_dict=feedtest_ctc)			
			#~ t2_2 = datetime.datetime.now()
			#~ print('test utterance: '+ str(minibatch * batsize_test)+'/'+str(test_size))
			#~ print('test accuracy: %.2f'% test_accuracy + 
			#~ " -- test time: " + str(t2_2-t2_1)[0:-7])
					
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(test_size)+'/'+str(test_size))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test error rate %.3f"%total_testacc)       
	t2_3 = datetime.datetime.now()
	test_time = t2_3-t2_1
	print("total test time: " + str(test_time)[0:-7]+'\n')		
	
	# info to start tensorboard:
	print('=============================================')
	print("tensorboard")
	print('=============================================')
	print("to start tensorboard via bash enter:")
	print("tensorboard --logdir " + tboard_path + tboard_name)
	print("open your Browser with:\nlocalhost:6006")
	
	testparams_list = [test_acc_history, test_filenames_history, 
					   test_softmax_list, randomint_test]
	
	return testparams_list, total_testacc, test_time, phoneme_labels
				
########################################################################
# start main: 
########################################################################			
#start training: 
train_acc_history, cost_history, crossval_history, train_time, total_trainacc = training(epochs,training_batches)

# start testing: full set
testparams_list, total_testacc, test_time, phoneme_labels  = testing(test_batches,batsize_test)
#testing(test_batches,batsize_test)
#~ # start testing: emission probs
#~ testparams_list2, total_testacc2, test_time2, phoneme_labels2 = testing(test_batches2,batsize_test2)

########################################################################
# plot settings:
########################################################################

# plot model summary:
plot_model_sum = "model_summary: "+str(modtype)+", layer_dim: "+str(modlayer)+\
                          ", hid_units: "+str(time_steps)+", hid_size: "+str(hidden_dim)+", frames: "+\
                          str(framenum)+", classes: " +str(classnum)+", epochs: "+str(epochs)+"\n"+\
                          bnorm+ ", "+ lnorm+", training accuracy %.3f"%total_trainacc+\
                          ", test accuracy %.3f"%total_testacc+" , "+\
                          str(timemarker)
                          
# define storage path for model:
model_path = path + "04_SpeechRNN/pretrained_RNN/"
model_name = str(timemarker)+"_"+modtype + "_"+ str(modlayer)+ "lay_"+ str(framenum)+ "fs_"+\
             str(coeffs)+ "coeffs_"+\
             str(time_steps)+ "hid_units_"+str(hidden_dim)+"hid_size_"+\
             str(classnum)+ "class_" +str(epochs)+"eps_"+\
             "_"+ bnorm+ "_"+ lnorm+\
             "_"+str(total_testacc)[0:-9]+"testacc"
print("Model saved to: ")
print(model_path + model_name)
                  
#init moving average:
wintrain = 300
wintest = 300
wtrain_len = len(cost_history)
if wtrain_len < 100000:
	wintrain = 5
wtest_len = len(testparams_list[0])
if wtest_len < 10000:
	wintest = 5
	
#=======================================================================
# plot training
#=======================================================================

# plot training loss function
fig1 = plt.figure(1, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='#1E90FF')
plt.plot(np.convolve(cost_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(cost_history),0,int(10)+1])
# plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batsize_train))
plt.ylabel('loss')

# plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_acc_history)),train_acc_history, color='#20B2AA')
plt.plot(np.convolve(train_acc_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(cost_history),0,1])
plt.title('Training Error')
plt.xlabel('number of batches, batch size:'+ str(batsize_train))
plt.ylabel('error percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
#pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

# plot validation loss function
plt.figure(2, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(crossval_history)),crossval_history, color='#1E90FF')
plt.plot(np.convolve(crossval_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(crossval_history),0,int(10)+1])
# plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

# plot test accuracy function
plt.subplot(212)
plt.plot(range(len(testparams_list[0])),testparams_list[0], color='m')
plt.plot(np.convolve(testparams_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(testparams_list[0]),0,1])
plt.title('Test Error')
plt.xlabel('number of batches, batch size: '+ str(batsize_test))
plt.ylabel('error percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')

########################################################################
# start export:
########################################################################

print('=============================================')
print("start export")
print('=============================================')

print("Model saved to: ")
print(model_path + model_name)

# fetch weights from tf.Graph
rnn_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="RNN_cell")
rnn_out = sess.run(rnn_outx)

W_cell = rnn_out[0]
W_out = np.array(sess.run(W_out))
b_cell = rnn_out[1]
b_out = np.array(sess.run(b_out))

# .mat file method:-----------------------------------------------------

# create dict model
model = {"model_name" : model_name}
# modelweights
model["W_cell"] = W_cell
model["W_out"] = W_out
model["b_cell"] = b_cell
model["b_out"] = b_out
model["acttype"] = acttype
# modellayout
model["hiddenlayer"] = modlayer
model["time steps"] = time_steps
model["hidden_dim"] = hidden_dim
model["classnum"] = classnum
model["framenum"] = framenum
model["randomint"] = randomint
model["classification"] = classifier 
model["optimizer"] = optimizing
model["batch_norm"] = bnorm
model["layer_norm"] = lnorm
# trainingparams
model["epochs"] = epochs
model["learnrate"] = learnrate
model["batsize_train"] = batsize_train
model["total_trainacc"] = total_trainacc
# testparams
model["batsize_test"] = batsize_test
model["randomint_test"] = testparams_list[3]
model["total_testacc"] = total_testacc
# history = [cost_history, train_acc_history, test_acc_history]
model["cost_history"] = cost_history
model["train_acc_history"] = train_acc_history
model["test_acc_history"] = testparams_list[0]

# from testing: files and emission probs
#model["test_filenames_history"] = testparams_list2[1]
#model["test_softmax_list"] = testparams_list2[2]
model["phoneme_labels"] = phoneme_labels
#modelw["label_transcript_list"] = label_transcript_list
#modelw["test_max_transcript_list"] = test_max_transcript_list

# save plot figures
model["fig_trainplot"] = im1
model["fig_testplot"] = im2

#export to .csv file
model_csv = [timemarker,total_trainacc,total_testacc,
             modtype,modlayer,framenum, coeffs, time_steps,classnum,
             classifier,epochs,acttype,learnrate,optimizing,
             bnorm,lnorm,batsize_train,batsize_test, 'No',hidden_dim]
df = pd.DataFrame(columns=model_csv)

#save only good models
if total_trainacc < 0.25:
	scipy.io.savemat(model_path + model_name,model)
	df.to_csv('nn_model_statistics/nn_model_statistics.csv', mode='a')
	df.to_csv('nn_model_statistics/nn_model_statistics_backup.csv', mode='a')
	
print('=============================================')
print("export finished")
print('=============================================')
print(" "+"\n")

# print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: "+ str(modtype)+" "+ str(modlayer)+" hidden layer")
print("*******************************************")
print("epochs: "+str(epochs))
print("hidden units: "+str(time_steps)+" each layer")
print("hidden dim: "+str(hidden_dim)+" each layer")
print("frame inputs: "+str(framenum))
print("optimizer: Adam")
print("-------------------------------------------")
print("data name: RVG new")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(train_time)[0:-7])
print("overall training error %.3f"%total_trainacc ) 
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_time)[0:-7])	
print("overall test error %.3f"%total_testacc) 
print("*******************************************")


# plot show options:----------------------------------------------------
plt.show()
#plt.hold(False)
#plt.close()

