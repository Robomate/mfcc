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

#https://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

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

from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

try:
	import cPickle as pickle
except:
   import _pickle as pickle

# remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

def create_minibatch(minibatchsize,time_steps,coeffs,filenames):
	'''create Minibatch for data and labels'''
	minibatch_data = np.zeros([minibatchsize,time_steps,coeffs])
	minibatch_label = np.zeros([minibatchsize,time_steps,classnum])
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
		# labels: one-hot-encoding
		for pos_lab in range(time_steps):
			label = int(label_padded[pos_lab])
			minibatch_label[batch][pos_lab][label-1] = 1.0					
	return minibatch_data, minibatch_label

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
		
########################################################################
# init parameter
########################################################################

print('=============================================')
print("load filenames")
print('=============================================\n')

#choose cell: ['RNN','BASIC_LSTM','LSTM', 'GRU', 'BiLSTM']
cell_number = 4

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
cell_name = ['RNN','BASIC_LSTM','LSTM', 'GRU', 'BiLSTM']
modtype = cell_name[cell_number]    # options: 'RNN', 'BASIC_LSTM'..
modlayer = 2                    # layer deepness of the network
coeffs = 39                      # MFCC: 39, DFT:257
time_steps = 1000                    # unit over time
hidden_dim = 200					# size of hidden layer
classnum = 135					# 3state mono: 135, triphone: 4643
framenum = 1 					    # number of input frames
inputnum = framenum * coeffs 		# length of input vector
randomint = 1
display_step = 100 					# to show progress
bnorm = 'no_bnorm'
lnorm = 'no_lnorm'
optimizing = 'adam'
classifier = 'softmax'
acttype = 'tanh'

# init training parameter:
epochs = 300                      
learnrate = 1e-4                        #high:1e-4
train_size  = len(trainset_name) 
val_size = len(valset_name)
tolerance = 0.01                        #break condition  
batsize_train = 64											
train_samples = int(train_size/batsize_train)

# init test parameter:
test_size = len(testset_name)	
batsize_test = 100
test_samples = int(test_size/batsize_test)    #number of test_samples

# init emission probs parameter	
batsize_test2 = 1
test_samples2 = 10

#~ #random shuffle filenames:
#~ valset_rand = random_shuffle_data(randomint, valset_name)
#~ #print(valset_rand)

# init params:		 
headerbytes = 12
datapath  = path + data_dir 
labelpath = path + label_dir

# init tboard
logdir = "tboard_logs/"	 
tboard_path = path + logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

# perform unit testing:
epochs = 1 
train_samples = 10
test_samples = 10

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

# weights: BiLSTM_FW layer
with tf.name_scope(name="BiLSTM_FW"):
	W_out_fw = weight_variable([hidden_dim, classnum])
	tf.summary.histogram("weights", W_out_fw)
# weights: BiLSTM_BW layer
with tf.name_scope(name="BiLSTM_BW"):
	W_out_bw = weight_variable([hidden_dim, classnum])
	tf.summary.histogram("weights", W_out_bw)
# weights: BiLSTM_bias
with tf.name_scope(name="BiLSTM_b"):
	b_out = bias_variable([classnum])
	tf.summary.histogram("biases", b_out)

# init placeholder:
x_input  = tf.placeholder(tf.float32,[None, time_steps, coeffs], name='inputs')  # ( batch,time, in)
y_target = tf.placeholder(tf.int32,[None, time_steps, classnum], name='labels') # ( batch,time, out)
# multi RNN cells
cells_fw = []
cells_bw = []
for _ in range(modlayer):
  cell_fw_init = tf.contrib.rnn.LSTMCell(hidden_dim,state_is_tuple=True)
  cell_bw_init = tf.contrib.rnn.LSTMCell(hidden_dim,state_is_tuple=True)
  cells_fw.append(cell_fw_init)
  cells_bw.append(cell_bw_init)
#stack cells into a list 
stacked_fw_cell = tf.contrib.rnn.MultiRNNCell(cells_fw,state_is_tuple=True)
stacked_bw_cell = tf.contrib.rnn.MultiRNNCell(cells_bw,state_is_tuple=True)
# BiLSTM cell
with tf.variable_scope("BiLSTM_cell"):
	rnn_outputs, _  = tf.nn.bidirectional_dynamic_rnn(stacked_fw_cell,stacked_bw_cell,x_input, dtype=tf.float32)
output_fw, output_bw = rnn_outputs

#~ # init cell variant: ['BASIC_RNN','RNN','BASIC_LSTM','LSTM', 'GRU']
#~ if cell_boolean[0]:
	#~ cell = tf.contrib.rnn.BasicRNNCell(hidden_dim)
#~ elif cell_boolean[1]:
	#~ cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, forget_bias=0.0, state_is_tuple=True)
#~ elif cell_boolean[2]:
	#~ cell = tf.contrib.rnn.LSTMCell(hidden_dim,state_is_tuple=True)
#~ elif cell_boolean[3]:
	#~ cell = tf.contrib.rnn.GRUCell(hidden_dim)
#~ else:
    #~ print("ERROR: wrong cell type!!")
#~ #print(cell)

########################################################################
#1. alternative:	
# reshape rnn_outputs
output_fw2 = tf.reshape(output_fw, [-1, hidden_dim])
output_bw2 = tf.reshape(output_bw, [-1, hidden_dim])
# linear output layer
output_fw3 = tf.matmul(output_fw2, W_out_fw)
output_bw3 = tf.matmul(output_bw2, W_out_bw)
output = tf.add(tf.add(output_fw3,output_bw3),b_out)
########################################################################
#~ #2. alternative via concat:
#~ outputsx = tf.concat(rnn_outputs, 2)
#~ outputs_conc = tf.reshape(outputsx, [-1, 2 * time_steps])
#~ # linear output layer
#~ output_bi_conc = tf.matmul(outputs_conc, weight_variable([2 * time_steps, classnum])) + bias_variable([classnum])
#~ #output_bi2 = tf.reshape(output_bi, [-1, batch_x_shape[0], n_hidden_6])
########################################################################

# reshape targets
y_reshaped = tf.reshape(y_target, [-1, classnum])
# get softmax probs
with tf.name_scope("softmax"):
	predictions = tf.nn.softmax(output)

# define loss, optimizer, accuracy:
with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y_reshaped))
	tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)
with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_reshaped,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", accuracy)

# compute metrics:
# inputs are binary tensors	
precision_tf = tf.contrib.metrics.streaming_precision(output, y_reshaped, name="precision")
recall_tf = tf.contrib.metrics.streaming_recall(output, y_reshaped, name="recall")
conf_matrix_tf = tf.confusion_matrix(tf.argmax(y_reshaped, 1), tf.argmax(output, 1), name="confusion_matrix")

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

def training(epochs,train_samples):
	'''train the neural model'''
	print('=============================================')
	print('start '+str(modtype)+' training')
	print('=============================================')
	t1_1 = datetime.datetime.now()
	
	# init cost, accuracy:
	crossval_history = np.empty(shape=[0],dtype=float)
	cost_history = np.empty(shape=[0],dtype=float)
	train_acc_history = np.empty(shape=[0],dtype=float)
	f1score_history = np.empty(shape=[0],dtype=float)

	# epoch loop:
	for epoch in range(1,epochs+1):
		
		# random shuffle filenames for each epoch:
		randomint_train = epoch
		trainset_rand = random_shuffle_data(randomint_train, trainset_name)
		
		# training loop: 
		for minibatch in range(int(train_samples)):
			# grab linear utterances from random trainset:
			trainset_buffer = trainset_rand[minibatch * batsize_train:(minibatch * batsize_train) + batsize_train]
			# minibatch_data [Batch Size, Sequence Length, Input Dimension]
			minibatch_train_data, minibatch_train_label = create_minibatch(batsize_train,time_steps,coeffs,trainset_buffer)
			# start feeding data into the model:
			feedtrain = {x_input: minibatch_train_data, y_target: minibatch_train_label }
			optimizer.run(feed_dict = feedtrain)
			
			# log history for tensorboard
			if minibatch % 5 == 0:
				[train_accuracy, s] = sess.run([accuracy, summ], feed_dict=feedtrain)
				# writer.add_summary(s, batch)
				
			# get cost_history, accuracy history data:
			cost_history = np.append(cost_history,sess.run(cross_entropy,feed_dict=feedtrain))
			train_acc_history = np.append(train_acc_history,sess.run(accuracy,feed_dict=feedtrain))
			
			# metrics
			#y_p = tf.argmax(output, 1)
			val_accuracy, y_pred, y_true  = sess.run([accuracy, tf.argmax(output, 1), tf.argmax(y_reshaped, 1)], feed_dict= feedtrain )
			#print ("validation accuracy:", val_accuracy)
			#~ print(y_pred) #output
			#~ print(y_pred.shape) #output
			#~ print(y_true) #labels
			#~ print(y_true.shape) #labels
			
			#precision = precision_score(y_true, y_pred,average="weighted")
			#print ("precision: "+str(precision)[0:-8])
			#recall = recall_score(y_true, y_pred,average="weighted")
			#print ("recall: "+ str(recall)[0:-7])
			f1score = f1_score(y_true, y_pred,average="weighted")
			f1score_history = np.append(f1score_history,f1score)
			#print ("f1_score: "+ str(f1score)[0:-8])
			
			#conf_matrix = confusion_matrix(y_true, y_pred)
			#print ("confusion_matrix", conf_matrix.shape)
			#print (conf_matrix[0:5,0:5])
			#print ("confusion_matrix"+ str(conf_matrix))
			
			#tf metrics:
			#~ print('sess.run(precision_tf, feed_dict= feedtrain)')
			#~ print(sess.run(conf_matrix_tf, feed_dict= feedtrain)[0:5,0:5])
			
			# fetch weights from tf.Graph
			rnn_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="BiLSTM_cell")
			rnn_out = sess.run(rnn_outx)
			
			#print weights:
			#~ print('sess.run(tf.get_collection')
			#~ print(len(rnn_out))
			#~ print(rnn_outx)
		
			
			#~ print('sess.run(rnn_outputs, feed_dict= feedtrain)')
			#~ print(sess.run(rnn_outputs, feed_dict= feedtrain)[0].shape)
			
			
			
			# check progress: training accuracy
			if  minibatch%10 == 0:
				train_accuracy = accuracy.eval(feed_dict=feedtrain)
				crossvalidationloss = sess.run(cross_entropy,feed_dict=feedtrain)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(minibatch * batsize_train)+'/'+str(train_size)+
				" -- cross validation loss: " + str(crossvalidationloss)[0:-2])
				print('training accuracy: %.2f'% train_accuracy + " -- F1 score: "+ str(f1score)[0:-10]+
				" -- training time: " + str(t1_2-t1_1)[0:-7])
				#print ("f1_score: "+ str(f1score)[0:-8])
				
			#~ #stopping condition:
			#~ if abs(crossval_history[-1] - crossval_history[-2]) < tolerance:
				#~ break
		print('=============================================')	
	print('=============================================')
	# total training statistics:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training accuracy %.3f"%total_trainacc)
	total_f1 = np.mean(f1score_history, axis=0)
	print("overall f1 score %.3f"%total_f1)        
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')	
	
	return train_acc_history, cost_history, crossval_history, train_time, total_trainacc,f1score_history,total_f1

########################################################################
# start testing:
########################################################################

def testing(test_samples=10,batsize_test=1, phase_bool=False, randomint_test=1):
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
	if test_samples < 20:
		print("phoneme_labels: ")
		print(phoneme_labels)
		print(" ")
		
	#random shuffle filenames:	
	testset_rand = random_shuffle_data(randomint_test, testset_name)
	
	
	# init histories
	test_softmax_utterance = np.empty(shape=[0,classnum],dtype=float)

	#test loop:
	for minibatch in range(int(test_samples)):
			# grab linear utterances from random trainset:
			testset_buffer = testset_rand[minibatch * batsize_test:(minibatch * batsize_test) + batsize_test]
			# minibatch_data [Batch Size, Sequence Length, Input Dimension]
			minibatch_test_data, minibatch_test_label = create_minibatch(batsize_test,time_steps,coeffs,testset_buffer)
			# start feeding data into the model:
			feedtest = {x_input: minibatch_test_data, y_target: minibatch_test_label}
			predictions = accuracy.eval(feed_dict = feedtest)
			test_acc_history = np.append(test_acc_history,predictions)
			
			# get label to phoneme transcription:
			#label_transcript = phonemeTranslator(minibatch_test_label, phoneme_labels)
		
			#check progress: test accuracy
			if  minibatch%10 == 0:
				test_accuracy = accuracy.eval(feed_dict=feedtest)			
				t2_2 = datetime.datetime.now()
				print('test utterance: '+ str(minibatch * batsize_test)+'/'+str(test_size))
				print('test accuracy: %.2f'% test_accuracy + 
				" -- test time: " + str(t2_2-t2_1)[0:-7])
					
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(test_size)+'/'+str(test_size))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test accuracy %.3f"%total_testacc)     
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
train_acc_history, cost_history, crossval_history, train_time, total_trainacc,f1score_history,total_f1 = training(epochs,train_samples)
# start testing: full set
testparams_list, total_testacc, test_time, phoneme_labels  = testing(test_samples,batsize_test)
# start testing: emission probs
testparams_list2, total_testacc2, test_time2, phoneme_labels2 = testing(test_samples2,batsize_test2)

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
#plt.plot(np.convolve(cost_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(cost_history),0,int(10)+1])
# plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batsize_train))
plt.ylabel('loss')

# plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_acc_history)),train_acc_history, color='#20B2AA')
#plt.plot(np.convolve(train_acc_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(cost_history),0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches, batch size:'+ str(batsize_train))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
#pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

# plot f1 score training function
plt.figure(2, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(f1score_history)),f1score_history, color='#1E90FF')
#plt.plot(np.convolve(f1score_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(f1score_history),0,int(10)+1])
# plt.axis([0,len(f1score_history),0,int(np.max(crossval_history))+1])
plt.title('F1_Score Training')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

# plot test accuracy function
plt.subplot(212)
plt.plot(range(len(testparams_list[0])),testparams_list[0], color='m')
#plt.plot(np.convolve(testparams_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(testparams_list[0]),0,1])
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batsize_test))
plt.ylabel('accuracy percentage')

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
rnn_outx = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope="BiLSTM_cell")
rnn_out = sess.run(rnn_outx)
# e.g. 3 layer: 12(36) weights+b, first 6 fw, second 6 bw

#print weights:
#~ print('sess.run(tf.get_collection')
#~ print(len(rnn_out))
#~ print(rnn_outx)

# BiLSTM cell
W_cell_fw = rnn_out[0]
b_cell_fw = rnn_out[1]
W_cell_bw = rnn_out[2]
b_cell_bw = rnn_out[3]
# output layer
W_out_fw = np.array(sess.run(W_out_fw))
W_out_bw = np.array(sess.run(W_out_bw))
b_out = np.array(sess.run(b_out))

# .mat file method:-----------------------------------------------------

# create dict model
model = {"model_name" : model_name}
# modelweights
model["W_cell_fw"] = W_cell_fw
model["b_cell_fw"] = b_cell_fw
model["W_cell_bw"] = W_cell_bw
model["b_cell_bw"] = b_cell_bw
model["W_out_fw"] = W_out_fw
model["W_out_bw"] = W_out_bw
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
model["total_f1_train"] = total_f1
# testparams
model["batsize_test"] = batsize_test
model["randomint_test"] = testparams_list[3]
model["total_testacc"] = total_testacc
# history = [cost_history, train_acc_history, test_acc_history]
model["cost_history"] = cost_history
model["train_acc_history"] = train_acc_history
model["test_acc_history"] = testparams_list[0]
model["f1score_history"] = f1score_history

# from testing: files and emission probs
model["test_filenames_history"] = testparams_list2[1]
model["test_softmax_list"] = testparams_list2[2]
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
if total_trainacc > 0.25:
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
print("overall training accuracy %.3f"%total_trainacc )
print("overall training f1 score %.3f"%total_f1 )
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_time)[0:-7])	
print("overall test accuracy %.3f"%total_testacc) 
print("*******************************************")


# plot show options:----------------------------------------------------
#plt.show()
#plt.hold(False)
#plt.close()

