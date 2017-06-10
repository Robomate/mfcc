#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#=======================================================================
#Purpose: Acoustic Model (Speech Recognition) 
#
#Model:   5 Hidden Layer MLP
#         2048 nodes each layer, trained with Adam
#		  ca. 17.6 Mio training parameter
#
#Inputs:  Bavarian Speech Corpus (German)
#		  training utterances: 		56127
#		  validation utterances: 	 7012
#         test utterances: 		 	 7023
#         --------------------------------
#		  total					    70162						
#
#		  shape of input: 1x273 vector (39MFCCcoeff * 7frames)
#
#Output:  135 classes (45 monophones with 3 HMM states each)
#Version: 4/2017 Roboball (MattK.)

#Start tensorboard via bash: 	tensorboard --logdir /logfile/directory
#Open Browser for tensorboard:  localhost:6006
#tensorboard --logdir /home/praktiku/korf/speechdata/tboard_logs/MLP_5layer_2017-05-10_14:04
#https://github.com/RuiShu/micro-projects/blob/master/tf-batchnorm-guide/batchnorm_guide.ipynb
#=======================================================================
'''

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

def padUtterance(framenum, labelfile, datafile):
	'''pad utterance on both sides'''
	if labelfile.shape[0] != datafile.shape[1]:
		labelfile = labelfile[0:-1]	
	#extract start and end vectors
	startveclabel = int(labelfile[0])
	startvecdata = datafile[:,0]
	endveclabel = int(labelfile[-1])
	endvecdata = datafile[:,-1]	
	#compose pads
	startmatlabel = np.tile(startveclabel,int((framenum-1)/2))
	startmatdata = np.tile(startvecdata,(int((framenum-1)/2),1)).T	
	endmatlabel = np.tile(endveclabel,int((framenum-1)/2))
	endmatdata = np.tile(endvecdata,(int((framenum-1)/2),1)).T	
	#pad utterance
	paddedlabel = np.concatenate((startmatlabel, labelfile, endmatlabel), axis=0)
	paddeddata = np.concatenate((startmatdata,datafile, endmatdata), axis=1)		
	return paddedlabel, paddeddata
	
def slidingWin(framenum, paddedlabel, paddeddata):
	'''create sliding windows to extract chunks'''
	#init: empty lists
	labellist=[]
	datalist=[]	
	#sliding window over utterance
	for pos in range(paddedlabel.shape[0]-framenum+1):
		slicelabel = paddedlabel[pos:pos+framenum]
		label = slicelabel[int((framenum-1)/2)]
		slicedata = paddeddata[:,pos:pos+framenum]
		slicedata_vec = np.reshape(slicedata, (coeffs*framenum), order='F') 	
		labellist.append(label)
		datalist.append(slicedata_vec)		
	return labellist, datalist
	
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
		
def createMinibatchData(minibatchsize, framenum, datavectorbuffer):
	'''create minibatches from databuffer'''
	minibatchdata = np.zeros(shape=(minibatchsize,framenum * coeffs))		
	#save vectors to minibatch array
	for vectorpos in range(minibatchsize):
		minibatchdata[vectorpos][:] = datavectorbuffer[vectorpos]
	return minibatchdata

def createMinibatchLabel(minibatchsize , classnum, labelbuffer):
	'''create one hot encoding from labels'''
	#init labels with zeros
	minibatchlabel = np.zeros(shape=(minibatchsize,classnum))	
	#one-hot-encoding
	for labelpos in range(minibatchsize):	
		label = int(labelbuffer[labelpos])
		minibatchlabel[labelpos][label-1] = 1.0
	return minibatchlabel	

def Preprocess(framenum, randbatnum, dataset_name, path):
	'''load data and label files, pad utterance, create chunks'''
	#init params:		 
	headerbytes = 12
	datapath  = path + "data/pattern_hghnr_39coef/"
	labelpath = path + "data/nn_output/"	
	#load data file:
	datafilename = datapath + dataset_name[randbatnum]
	datafile = loadPatData(datafilename, headerbytes)	
	#load label file:
	dataset_namelab = dataset_name[randbatnum][:-4]+".txt"
	inputlabel = labelpath + dataset_namelab
	labelfile = np.loadtxt(inputlabel) #read in as float
	#pad utterance:
	paddedlabel, paddeddata = padUtterance(framenum, labelfile, datafile)	
	#extract vector and labels by sliding window:
	labellist, datavectorlist = slidingWin(framenum, paddedlabel, paddeddata)	
	return labellist, datavectorlist

def Databuffer(buffersamples, framenum, dataset_name, path):
	'''create buffer for minibatch'''
	#init buffer
	labelbuffer = []
	datavectorbuffer = []	
	for pos in range(buffersamples):		
		labellist, datavectorlist = Preprocess(framenum, pos, dataset_name, path)
		labelbuffer = labelbuffer+labellist
		datavectorbuffer = datavectorbuffer+datavectorlist
	return labelbuffer, datavectorbuffer

def phonemeDict():
	'''Dictionary for classes'''
	phoneme_dict = ["@","a","a:","aI","an","ar","aU","b","C","d","e:","E",
	"E:","f","g","h","i:","I","j","k","l","m","n","N","o:","O","Oe","On",
	"OY","p","r","s","S","sil","sp","t","u","u:","U","v","x","y:","Y","z","Z"]
	#create label translation
	phoneme_lab = np.arange(135).astype('str')
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

# fix initial randomness:
randomint = 1
modtype = "MLP" 
modlayer = 5	 	

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
pathname = "data/dataset_filenames/"
logdir = "tboard_logs/"
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
nodes = 2*512 							#size of hidden layer
classnum = 135 							#number of output classes
framenum = 11 							#number of frames
inputnum = framenum * coeffs 				#length of input vector
display_step = 100 						#to show progress
bnorm = 'no_bnorm'

# error message frame number:
if framenum%2==0:
	print("Error, please choose uneven number of frames.")
	
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
#random shuffle weights inits:
rand_w = createRandomvec(randomint, modlayer+1)

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
		 
def activfunction(x, acttype):
	'''activation functions'''
	if acttype == "tanh":
		activation_array = tf.tanh(x)
	elif acttype == "relu":
		activation_array = tf.nn.relu(x)
	else:
		activation_array = tf.sigmoid(x)
	return activation_array

# init activation function:
actdict = ["tanh", "relu", "sigmoid"]
acttype = actdict[0]

# init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None, inputnum],name="input")
y_target = tf.placeholder(tf.float32, shape=[None, classnum],name="labels")

# init weights:	
# 1. hidden layer
with tf.name_scope(name="fc1"):
	W1 = weight_variable([inputnum, nodes]) 
	b1 = bias_variable([nodes])
	tf.summary.histogram("weights", W1)
	tf.summary.histogram("biases", b1) 
# 2. hidden layer 
with tf.name_scope(name="fc2"):
	W2 = weight_variable([nodes, nodes]) 
	b2 = bias_variable([nodes])
	tf.summary.histogram("weights", W2)
	tf.summary.histogram("biases", b2)  
# 3. hidden layer
with tf.name_scope(name="fc3"):
	W3 = weight_variable([nodes, nodes]) 
	b3 = bias_variable([nodes])
	tf.summary.histogram("weights", W3)
	tf.summary.histogram("biases", b3) 
# 4. hidden layer
with tf.name_scope(name="fc4"):
	W4 = weight_variable([nodes, nodes]) 
	b4 = bias_variable([nodes])
	tf.summary.histogram("weights", W4)
	tf.summary.histogram("biases", b4)  
# 5. hidden layer
with tf.name_scope(name="fc5"):
	W5 = weight_variable([nodes, nodes])
	b5 = bias_variable([nodes])
	tf.summary.histogram("weights", W5)
	tf.summary.histogram("biases", b5)
# 6. output layer
with tf.name_scope(name="fc6"):
	W6 = weight_variable([nodes, classnum])
	b6 = bias_variable([classnum])
	tf.summary.histogram("weights", W6)
	tf.summary.histogram("biases", b6)

# define model:
layer_1 = activfunction(matmul(x_input,W1,b1,name="fc1"),acttype)
layer_2 = activfunction(matmul(layer_1,W2,b2,name="fc2"),acttype)
layer_3 = activfunction(matmul(layer_2,W3,b3,name="fc3"),acttype)
layer_4 = activfunction(matmul(layer_3,W4,b4,name="fc4"),acttype) 
layer_5 = activfunction(matmul(layer_4,W5,b5,name="fc5"),acttype)
layer_6 = dense(layer_5, W6, b6)

# define classifier, cost function:
softmax = tf.nn.softmax(layer_6)

# define loss, optimizer, accuracy:
with tf.name_scope("cross_entropy"):
	cross_entropy = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(logits= layer_6,labels = y_target))
	tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope("train"):
	optimizer = tf.train.AdamOptimizer(learnrate).minimize(cross_entropy)
with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(layer_6,1), tf.argmax(y_target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", accuracy)

# merge all summaries for tensorboard:	
summ = tf.summary.merge_all()

# init tf session :
sess = tf.InteractiveSession()
# save and restore all the variables:
saver = tf.train.Saver()
# start session:
sess.run(tf.global_variables_initializer()) 
# init tensorboard
writer = tf.summary.FileWriter(tboard_path + tboard_name)
writer.add_graph(sess.graph)

# print out model info:
print("**********************************")
print("model: 5 hidden layer MLP")
print("**********************************")
print("hidden units: "+str(nodes)+" each layer")
print("activation function: "+str(acttype))
print("optimizer: Adam")
print("----------------------------------")
print("data name: Bavarian Speech Corpus")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size)+"\n")

########################################################################
# training loop:
########################################################################

def training(epochs=1,train_samples=10):
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
		
		#random shuffle filenames for each epoch:
		randomint_train = epoch
		trainset_rand = randomShuffleData(randomint_train, trainset_name)
		
		#training loop: length = int(train_size/buffersamples)	
		for buffernum in range(train_samples):
			
			#grab linear utterances (buffersamples) from random trainset:
			trainset_buffernum = trainset_rand[buffernum * buffer_train:(buffernum * buffer_train) + buffer_train]
			
			#create buffer:
			labelbuffer, datavectorbuffer = Databuffer(buffer_train, framenum, trainset_buffernum, path)
			bufferlength = len(labelbuffer)
			#~ print("buffer number: "+str(buffernum))
			#~ print("buffer length: "+str(bufferlength))
			
			#get number of minibatches from buffer:
			minibatches = int(bufferlength/batsize_train)
			#~ print("number of minibatches: "+str(minibatches))
			
			#minibatch loop per buffer:
			for batch in range(minibatches):
				
				#grab linear minibatch datavectors from databuffer:
				datavector = datavectorbuffer[batch * batsize_train: (batch * batsize_train)+batsize_train]		
				#grab linear minibatch labels from labelbuffer:		
				labelvector = labelbuffer[batch * batsize_train: (batch * batsize_train)+batsize_train]
							
				#minibatch data:
				trainbatchdata = createMinibatchData(batsize_train, framenum, datavector)
				#minibatch labels: one hot encoding
				trainbatchlabel = createMinibatchLabel(batsize_train, classnum, labelvector)
						
				#start feeding data into the model:
				feedtrain = {x_input: trainbatchdata, y_target: trainbatchlabel}
				optimizer.run(feed_dict = feedtrain)
				
				#log history for tensorboard
				if batch % 5 == 0:
					[train_accuracy, s] = sess.run([accuracy, summ], feed_dict=feedtrain)
					writer.add_summary(s, batch)
					
				#get cost_history, accuracy history data:
				cost_history = np.append(cost_history,sess.run(cross_entropy,feed_dict=feedtrain))
				train_acc_history = np.append(train_acc_history,sess.run(accuracy,feed_dict=feedtrain))
			
			#check progress: training accuracy
			if  buffernum%10 == 0:
				train_accuracy = accuracy.eval(feed_dict=feedtrain)
				crossvalidationloss = sess.run(cross_entropy,feed_dict=feedtrain)
				crossval_history = np.append(crossval_history,crossvalidationloss)
				t1_2 = datetime.datetime.now()
				print('epoch: '+ str(epoch)+'/'+str(epochs)+
				' -- training utterance: '+ str(buffernum * buffer_train)+'/'+str(train_size)+
				" -- cross validation loss: " + str(crossvalidationloss)[0:-2])
				print('training accuracy: %.2f'% train_accuracy + 
				" -- training time: " + str(t1_2-t1_1)[0:-7])
				
			#~ #stopping condition:
			#~ if abs(crossval_history[-1] - crossval_history[-2]) < tolerance:
				#~ break
		print('=============================================')
	
	print('=============================================')
	#Total Training Stats:
	total_trainacc = np.mean(train_acc_history, axis=0)
	print("overall training accuracy %.3f"%total_trainacc)       
	t1_3 = datetime.datetime.now()
	train_time = t1_3-t1_1
	print("total training time: " + str(train_time)[0:-7]+'\n')	
	
	return train_acc_history, cost_history, crossval_history, train_time, total_trainacc
	
	
########################################################################
#start testing:
########################################################################

def testing(test_samples=10,batsize_test=1,buffer_test=10, randomint_test=1):
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
	testset_rand = randomShuffleData(randomint_test, testset_name)
	
	#test loop: length = int(test_size/buffersamples)
	for buffernum in range(int(test_samples)):
			
			#grab linear minibatch datavectors from databuffer:
			testset_buffernum = testset_rand[buffernum * buffer_test: (buffernum * buffer_test)+buffer_test]
			test_filenames_history = np.append(test_filenames_history,testset_buffernum)
			
			#create buffer:
			labelbuffer, datavectorbuffer = Databuffer(buffer_test, framenum, testset_buffernum, path)	
			bufferlength = len(labelbuffer)
			#~ print("buffernum: "+str(buffernum))
			#~ print("bufferlength: "+str(bufferlength))
			
			#get label to phoneme transcription:
			label_transcript = phonemeTranslator(labelbuffer, phoneme_labels)
			
			#get number of minibatches from buffer:
			minibatches = int(bufferlength/batsize_test)
			#~ print("number of minibatches: "+str(minibatches))
			
			#init histories
			test_softmax_utterance = np.empty(shape=[0,135],dtype=float)
			
			#minibatch loop per buffer:
			for batch in range(minibatches):
				
				#grab minibatch datavectors from databuffer:
				datavector = datavectorbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]		
				#grab minibatch labels from labelbuffer:		
				labelvector = labelbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]
							
				#minibatch data:
				testbatchdata = createMinibatchData(batsize_test, framenum, datavector)
				#minibatch labels: one hot encoding
				testbatchlabel = createMinibatchLabel(batsize_test, classnum, labelvector)
				
				#start feeding data into the model:
				feedtest = {x_input: testbatchdata, y_target: testbatchlabel}
				predictions = accuracy.eval(feed_dict = feedtest)
				test_acc_history = np.append(test_acc_history,predictions)
				
				if test_samples < 20:
					#get emissionprobs
					softmaxprobs = sess.run(softmax,feed_dict=feedtest)
					test_softmax_utterance = np.append(test_softmax_utterance,softmaxprobs, axis=0)
			
			if test_samples < 20:
				#save emission-probs for each utterance
				test_softmax_list.append(test_softmax_utterance)
				
				#translation to phonemes
				testmax_loc = test_softmax_utterance.argmax(axis=1)
				test_max_transcript = np.zeros_like(testmax_loc).astype(str)
				for loc in range(len(testmax_loc)):
					test_max_transcript[loc] = phoneme_labels[testmax_loc[loc]]
				test_max_transcript_list.append(test_max_transcript)
				label_transcript_list.append(label_transcript)
				print('=============================================')
				print("test utterance: " + str(buffernum+1))
				print("label name: " + testset_buffernum[0]+"\n")
				print("label_transcript: ")
				print(str(label_transcript))
				print('test_max_transcript: ')
				print(str(test_max_transcript)+"\n")
		
			#check progress: test accuracy
			if  buffernum%10 == 0:
				test_accuracy = accuracy.eval(feed_dict=feedtest)			
				t2_2 = datetime.datetime.now()
				print('test utterance: '+ str(buffernum*buffer_test)+'/'+str(test_size))
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
#start main: 
########################################################################

#start training: 
train_acc_history, cost_history, crossval_history, train_time, total_trainacc = training(epochs,train_samples)
# start testing: full set
testparams_list, total_testacc, test_time, phoneme_labels  = testing(test_samples,batsize_test,buffer_test)
# start testing: emission probs
testparams_list2, total_testacc2, test_time2, phoneme_labels = testing(test_samples2,batsize_test2,buffer_test2)


########################################################################
#plot settings:
########################################################################

#plot model summary:
plot_model_sum = "model_summary: "+str(modtype)+"_"+str(modlayer)+" hidden layer, "+\
                          " hidden units: "+str(nodes)+" ,"+" frame inputs: "+\
                          str(framenum)+"\n"+\
                          "overall training accuracy %.3f"%total_trainacc+\
                          " , overall test accuracy %.3f"%total_testacc+" , "+\
                          str(timemarker)
                          
#define storage path for model:
model_path = path + "SpeechMLP/pretrained_MLP/"
model_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)+"_" + str(total_testacc)[0:-9]
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
#plot training
#=======================================================================

#plot training loss function
fig1 = plt.figure(1, figsize=(8,8))
plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
plt.subplots_adjust(wspace=0.5,hspace=0.5)
plt.subplot(211)
plt.plot(range(len(cost_history)),cost_history, color='#1E90FF')
plt.plot(np.convolve(cost_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
plt.title('Cross Entropy Training Loss')
plt.xlabel('number of batches, batch size: '+ str(batsize_train))
plt.ylabel('loss')

#plot training accuracy function
plt.subplot(212)
plt.plot(range(len(train_acc_history)),train_acc_history, color='#20B2AA')
plt.plot(np.convolve(train_acc_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
plt.axis([0,len(cost_history),0,1])
plt.title('Training Accuracy')
plt.xlabel('number of batches, batch size:'+ str(batsize_train))
plt.ylabel('accuracy percentage')

#export figure
img = plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
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
plt.plot(range(len(crossval_history)),crossval_history, color='#1E90FF')
plt.plot(np.convolve(crossval_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
plt.title('Cross Validation Loss')
plt.xlabel('number of validation checks')
plt.ylabel('loss')

#plot test accuracy function
plt.subplot(212)
plt.plot(range(len(testparams_list[0])),testparams_list[0], color='m')
plt.plot(np.convolve(testparams_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
plt.axis([0,len(testparams_list[0]),0,1])
plt.title('Test Accuracy')
plt.xlabel('number of batches, batch size: '+ str(batsize_test))
plt.ylabel('accuracy percentage')

#export figure
plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')
print(np.info(im2))


########################################################################
#start export:
########################################################################

print('=============================================')
print("start export")
print('=============================================')

print("Model saved to: ")
print(model_path + model_name)

W1 = np.array(sess.run(W1))
W2 = np.array(sess.run(W2))
W3 = np.array(sess.run(W3))
W4 = np.array(sess.run(W4))
W5 = np.array(sess.run(W5))
W6 = np.array(sess.run(W6))
 
b1 = np.array(sess.run(b1))
b2 = np.array(sess.run(b2))
b3 = np.array(sess.run(b3))
b4 = np.array(sess.run(b4))
b5 = np.array(sess.run(b5))
b6 = np.array(sess.run(b6))

 
#tensorflow method:-----------------------------------------------------
#save_path = saver.save(sess, model_path + model_name+".ckpt" )
#print("Model saved to file: %s" % save_path)

#~ #pickle method:---------------------------------------------------------
#~ #create export lists:
#~ modelweights = [W1, W2, W3, W4, W5, W6, b1, b2, b3, b4,b5, b6 ];
#~ modellayout = [model_name,nodes,classnum,framenum,acttype,randomint];
#~ trainingparams = [epochs, learnrate,batsize_train]
#~ testparams = [batsize_test, testparams_list[3]]
#~ history = [cost_history, train_acc_history, testparams_list[0]]
#~ testhistory = [testparams_list[1], testparams_list[2]]
#~ 
#~ #export trained model:
#~ f = open(model_path + model_name, "wb")
#~ pickle.dump(modelweights, f,protocol=2)
#~ pickle.dump(modellayout, f,protocol=2)
#~ pickle.dump(trainingparams, f,protocol=2)
#~ pickle.dump(testparams, f,protocol=2)
#~ pickle.dump(history, f,protocol=2)
#~ pickle.dump(testhistory, f,protocol=2)
#~ f.close()

#.mat file method:------------------------------------------------------

#create dict model
model = {"model_name" : model_name}
#modelweights
model["W1"] = W1
model["W2"] = W2
model["W3"] = W3
model["W4"] = W4
model["W5"] = W5
model["W6"] = W6
model["b1"] = b1
model["b2"] = b2
model["b3"] = b3
model["b4"] = b4
model["b5"] = b5
model["b6"] = b6
model["acttype1"] = acttype
model["acttype2"] = acttype
model["acttype3"] = acttype
model["acttype4"] = acttype
model["acttype5"] = acttype
#modellayout
model["hiddenlayer"] = modlayer
model["nodes"] = nodes
model["classnum"] = classnum
model["framenum"] = framenum
model["randomint"] = randomint
model["classification"] = 'softmax'
model["optimizer"] = 'adam'
#trainingparams
model["epochs"] = epochs
model["learnrate"] = learnrate
model["batsize_train"] = batsize_train
model["total_trainacc"] = total_trainacc
#testparams
model["batsize_test"] = batsize_test
model["randomint_test"] = testparams_list[3]
model["total_testacc"] = total_testacc
#history = [cost_history, train_acc_history, test_acc_history]
model["cost_history"] = cost_history
model["train_acc_history"] = train_acc_history
model["test_acc_history"] = testparams_list[0]

#from testing: files and emission probs
model["test_filenames_history"] = testparams_list2[1]
model["test_softmax_list"] = testparams_list2[2]
model["batch_norm"] = bnorm
model["phoneme_labels"] = phoneme_labels
#modelw["label_transcript_list"] = label_transcript_list
#modelw["test_max_transcript_list"] = test_max_transcript_list

#save plot figures
model["fig_trainplot"] = im1
model["fig_testplot"] = im2

scipy.io.savemat(model_path + model_name,model)

print('=============================================')
print("export finished")
print('=============================================')
print(" "+"\n")

#print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: "+ str(modtype)+"_"+ str(modlayer)+" hidden layer")
print("*******************************************")
print("hidden units: "+str(nodes)+" each layer")
print("frame inputs: "+str(framenum))
print("activation function: "+str(acttype))
print("optimizer: Adam")
print("-------------------------------------------")
print("data name: Bavarian Speech Corpus")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size))
print("-------------------------------------------")
print(str(modtype)+' training:')
print("total training time: " + str(train_time)[0:-7])
print("overall training accuracy %.3f"%total_trainacc ) 
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(test_time)[0:-7])	
print("overall test accuracy %.3f"%total_testacc) 
print("*******************************************")


#plot show options:----------------------------------------------------------
plt.show()
#plt.hold(False)
#plt.close()




