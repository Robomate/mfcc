#!/usr/bin/env python
# -*- coding: utf-8 -*-

#=======================================================================
#Purpose: Load Pretrained Acoustic Model (Speech Recognition) 
#
#Model:   5 Hidden Layer MLP
#         2048 nodes each layer, trained with Adam
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
#=======================================================================

import numpy as np
import tensorflow as tf
import random
import re
import datetime 
import matplotlib.pyplot as plt 
import os
import scipy.io
import h5py
try:
	import cPickle as pickle
except:
   import _pickle as pickle

#remove warnings from tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 

########################################################################
#define functions:
########################################################################

def loadNamefiles(filenames):
	'''load dataset-names from .txt files:'''
	#load dataset-names from .txt files:
	return np.genfromtxt(filenames, delimiter=" ",dtype='str')
		
def loadPatData(filename, headerbytes):
	'''load Pattern files: 39MFCC coefficients, 12byte header'''
	with open(filename, 'rb') as fid:
		frames = np.fromfile(fid, dtype=np.int32) #get frames
		#print (frames[0])
		fid.seek(headerbytes, os.SEEK_SET)  # read in without header offset
		datafile = np.fromfile(fid, dtype=np.float32).reshape((frames[0], 39)).T 
	return datafile

def padUtterance(framenum, labelfile, datafile):
	'''pad utterance on both sides'''
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
		slicedata_vec = np.reshape(slicedata, (39*framenum), order='F') 	
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
	minibatchdata = np.zeros(shape=(minibatchsize,framenum * 39))		
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
#import filenames: 
########################################################################

print('=============================================')
print("load filenames")
print('=============================================\n')

#fix inintial randomness:
randomint = 1 
modtype = "MLP" 
modlayer = 5		

#get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

#init paths:
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
path = "/home/korf/Desktop/BA_05/" #on lynx5
dir_filenames = "data/dataset_filenames/"
dir_data = "data/pattern_hghnr_39coef/"
dir_label = "data/nn_output/"
data_path = path + dir_data
label_path = path + dir_label

tboard_logdir   = "runlogs/tboard_logs/"
tboard_path = path + tboard_logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

#init filenames:
trainset_filename = 'trainingset.txt'
validationset_filename = 'validationset.txt'
testset_filename = 'testset.txt'
	
#load filenames:
trainset_name = loadNamefiles(path + dir_filenames + trainset_filename)
valset_name = loadNamefiles(path + dir_filenames + validationset_filename)
testset_name = loadNamefiles(path + dir_filenames + testset_filename)

#random shuffle filenames:
valset_rand = randomShuffleData(randomint, valset_name)

########################################################################
#import pretrained model data:
########################################################################

print('=============================================')
print("load pretrained model")
print('=============================================')

#define path for pretrained model:
model_path = path + "SpeechMLP/pretrained_MLP/pretrained_MLP/"
#model_name = "MLP_5layer_2017-05-10_16:39_0.497"
model_name = "MLP_5layer_2017-05-20_15:44_0.560"
print("Model loaded from: ")
print(model_path + model_name)

'''
#load pickle lists with data:
modelweights = [W1, W2, W3, W4, W5, W6, b1, b2, b3, b4,b5, b6 ];
modellayout = [nodes, classnum, framenum, acttype];
trainingparams = [randomint, epochs, learnrate,batsize_train, batsize_test]
history = [cost_history, train_acc_history, test_acc_history]
'''

#import pretrained model:
f = open(model_path + model_name, "rb")
modelweights = pickle.load(f)
modellayout = pickle.load(f)
trainingparams = pickle.load(f)
history = pickle.load(f)
f.close()

#print out info
#print(modelweights[0])
#print(modelweights[0].shape)
print('modellayout: '+ str(modellayout))
print('trainingparams: '+ str(trainingparams))
#print(history[2].shape)


print('=============================================')
print("loading finished")
print('=============================================')
print(" "+"\n")


########################################################################
#plot settings:
########################################################################

#=======================================================================
#plot pretrained testing
#=======================================================================

#~ plt.figure(2, figsize=(10,10))

#~ #plot training accuracy function
#~ plt.subplots_adjust(wspace=0.5,hspace=0.5)
#~ plt.subplot(311)
#~ plt.plot(range(len(history[0])),history[0], color='#B298DC')
#~ winsize=300
#~ #'aquamarine': '#7FFFD4','cadetblue':'#5F9EA0'
#~ plt.plot(np.convolve(history[0], np.ones((winsize,))/winsize, mode='valid'), color='#FFD7F9', alpha=1)
#~ plt.axis([0,len(history[1]),0,max(history[0])])
#~ plt.title('Pretrained Model: Training Loss')
#~ plt.xlabel('number of batches'+ str(trainingparams[3]))
#~ plt.ylabel('cross entropy loss')

#~ #plot training accuracy function
#~ plt.subplots_adjust(wspace=0.5,hspace=0.5)
#~ plt.subplot(312)
#~ plt.plot(range(len(history[1])),history[1], color='cadetblue')
#~ winsize=300
#~ #'aquamarine': '#7FFFD4','cadetblue':'#5F9EA0'
#~ plt.plot(np.convolve(history[1], np.ones((winsize,))/winsize, mode='valid'), color='aquamarine', alpha=1)
#~ plt.axis([0,len(history[1]),0,1])
#~ plt.title('Pretrained Model: Training Accuracy')
#~ plt.xlabel('number of batches, batch size: '+ str(trainingparams[3]))
#~ plt.ylabel('accuracy percentage')

#~ #plot test accuracy function
#~ plt.subplots_adjust(wspace=0.5,hspace=0.5)
#~ plt.subplot(313)
#~ plt.plot(range(len(history[2])),history[2], color='#FF9F6A')
#~ winsize=200
#~ #'aquamarine': '#7FFFD4','cadetblue':'#5F9EA0'
#~ plt.plot(np.convolve(history[2], np.ones((winsize,))/winsize, mode='valid'), color='#FFE6D1', alpha=1)
#~ plt.axis([0,len(history[2]),0,1])
#~ plt.title('Pretrained Model: Test Accuracy')
#~ plt.xlabel('number of batches, batch size: '+ str(trainingparams[4]))
#~ plt.ylabel('accuracy percentage')

#~ #plt.ion()
#~ plt.show()
#~ #plt.hold(False)
	
########################################################################
#init parameter
########################################################################

#define hyperparameter:
test_size = len(testset_name) 
						
batsize_test = 1							
batches_test = int(test_size / batsize_test) 	#round down
buffersamples = 1 		#choose number utterances in buffer

nodes    = modellayout[0] 				#size of hidden layer
classnum = modellayout[1]  				#number of output classes
framenum = modellayout[2]  				#number of frames
inputnum = framenum * 39 				#length of input vector
display_step = 100 						#to show progress

#init accuracy:
test_acc_history = np.empty(shape=[0],dtype=float)
test_softmax_utterance = np.empty(shape=[0,135],dtype=float)
test_filenames_history = np.empty(shape=[0],dtype=str)
test_softmax_list = []

########################################################################
#init and define model: MLP model
########################################################################

#init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None, inputnum],name="input")
y_target = tf.placeholder(tf.float32, shape=[None, classnum],name="labels")

#init model:
def matmul(x, W, b, name="fc"):
	with tf.name_scope(name):
		return tf.add(tf.matmul(x, W), b)
  
def activfunction(x, acttype):
	if acttype == "tanh":
		activation_array = tf.tanh(x)
	elif acttype == "relu":
		activation_array = tf.nn.relu(x)
	else:
		activation_array = tf.sigmoid(x)
	return activation_array

#init activation function:
actdict = ["tanh", "relu", "sigmoid"]
acttype = actdict[0]

#init weights:

#1. hidden layer
with tf.name_scope(name="fc1"):
	W1 = modelweights[0]
	b1 = modelweights[6]
	tf.summary.histogram("weights", W1)
	tf.summary.histogram("biases", b1) 
#2. hidden layer 
with tf.name_scope(name="fc2"):
	W2 = modelweights[1]
	b2 = modelweights[7]
	tf.summary.histogram("weights", W2)
	tf.summary.histogram("biases", b2)  
#3. hidden layer
with tf.name_scope(name="fc3"):
	W3 = modelweights[2]
	b3 = modelweights[8]
	tf.summary.histogram("weights", W3)
	tf.summary.histogram("biases", b3) 
#4. hidden layer
with tf.name_scope(name="fc4"):
	W4 = modelweights[3]
	b4 = modelweights[9]
	tf.summary.histogram("weights", W4)
	tf.summary.histogram("biases", b4)  
#5. hidden layer
with tf.name_scope(name="fc5"):
	W5 = modelweights[4]
	b5 = modelweights[10]
	tf.summary.histogram("weights", W5)
	tf.summary.histogram("biases", b5)
#6. output layer
with tf.name_scope(name="fc6"):
	W6 = modelweights[5]
	b6 = modelweights[11]
	tf.summary.histogram("weights", W6)
	tf.summary.histogram("biases", b6)


#define model: 5 hidden layer model
layer_1 = activfunction(matmul(x_input, W1, b1,name="fc1"),acttype)
layer_2 = activfunction(matmul(layer_1, W2, b2,name="fc2"),acttype)
layer_3 = activfunction(matmul(layer_2, W3, b3,name="fc3"),acttype)
layer_4 = activfunction(matmul(layer_3, W4, b4,name="fc4"),acttype) 
layer_5 = activfunction(matmul(layer_4, W5, b5,name="fc5"),acttype)
layer_6 = matmul(layer_5, W6, b6)

#define classifier, cost function:
softmax = tf.nn.softmax(layer_6)

#define accuracy:
with tf.name_scope("accuracy"):
	correct_prediction = tf.equal(tf.argmax(layer_6,1), tf.argmax(y_target,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	tf.summary.scalar("accuracy", accuracy)


#merge all summaries for tensorboard:	
summ = tf.summary.merge_all()

#init tf session :
sess = tf.InteractiveSession()
#start session:
sess.run(tf.global_variables_initializer()) 
#init tensorboard
writer = tf.summary.FileWriter(tboard_path + tboard_name)
writer.add_graph(sess.graph)

#print out model info:
print('=============================================')
print("              model testing                  ")
print('=============================================')
print("model: " + str(modlayer) + " hidden layer " + str(modtype))
print("hidden units: " + str(nodes) + " each layer")
print("activation function: "+ str(acttype))
print("optimizer: Adam")
print("-------------------------------------------")
print("data name: Bavarian Speech Corpus")
print("test data: " +str(test_size)+"\n")
 

########################################################################
#start testing:
########################################################################

print('=============================================')
print('start '+str(modtype)+' testing')
print('=============================================')
t2_1 = datetime.datetime.now()

#random shuffle filenames:
randomint_test = 1
testset_rand = randomShuffleData(randomint_test, testset_name)

#get label-dictionary:
phoneme_labels = phonemeDict()
print(phoneme_labels)

#test loop: length = int(test_size/buffersamples)
for buffernum2 in range(int(10)):
		
	#grab linear minibatch datavectors from databuffer:
	testset_buffernum = testset_rand[buffernum2 * buffersamples: (buffernum2 * buffersamples)+buffersamples]
	test_filenames_history = np.append(test_filenames_history,testset_buffernum)
	
	#create buffer:
	labelbuffer, datavectorbuffer = Databuffer(buffersamples, framenum, testset_buffernum, path)	
	bufferlength = len(labelbuffer)
	
	#get label to phoneme transcription:
	label_transcript = phonemeTranslator(labelbuffer, phoneme_labels)
	
	
	print("test utterance: "+str(buffernum2+1))
	print("bufferlength: "+str(bufferlength))
	
	#get number of minibatches from buffer:
	minibatches = int(bufferlength/batsize_test)
	print("number of minibatches: "+str(minibatches))
	
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
					
		############################################################
		#feed minibatch into model:
		############################################################
		
		#start feeding data into the model:
		feedtest = {x_input: testbatchdata, y_target: testbatchlabel}
		predictions = accuracy.eval(feed_dict = feedtest)
		test_acc_history = np.append(test_acc_history,predictions)
		#get emission-probs
		softmaxprobs = sess.run(softmax,feed_dict=feedtest)
		test_softmax_utterance = np.append(test_softmax_utterance,softmaxprobs, axis=0)
	#save emission-probs for each utterance
	test_softmax_list.append(test_softmax_utterance)

	
		
	#check progress: test accuracy
	if  buffernum2%10 == 0:
		test_accuracy = accuracy.eval(feed_dict=feedtest)			
		t2_2 = datetime.datetime.now()
		print('test utterance: '+ str(buffernum2*buffersamples)+'/'+str(test_size))
		print('test accuracy: %.2f'% test_accuracy + 
		" -- test time: " + str(t2_2-t2_1)[0:-7])
		
				
print('=============================================')
print('test utterance: '+ str(test_size)+'/'+str(test_size))
total_testacc = np.mean(test_acc_history, axis=0)
print("overall test accuracy %.3f"%total_testacc) 
			
#Measure Training Time:-------------------------------------------------       
t2_3 = datetime.datetime.now()
print("total test time: " + str(t2_3-t2_1)[0:-7]+'\n')		

#info to start tensorboard:
print('=============================================')
print("tensorboard")
print('=============================================')
print("to start tensorboard via bash enter:")
print("tensorboard --logdir " + tboard_path + tboard_name)
print("open your Browser with:\nlocalhost:6006")

#print out model summary:
print('=============================================')
print("model summary")
print('=============================================')
print("*******************************************")
print("model: " +str(modlayer)+" hidden layer "+str(modtype))
print("*******************************************")
print("hidden units: "+str(nodes)+" each layer")
print("activation function: "+str(acttype))
print("optimizer: Adam")
print("-------------------------------------------")
print("data name: Bavarian Speech Corpus")
print("test data: " +str(test_size))
print("-------------------------------------------")
print(str(modtype)+' testing:')
print("total test time: " + str(t2_3-t2_1)[0:-4])	
print("overall test accuracy %.3f"%total_testacc) 
print("*******************************************")

########################################################################
#start export:
########################################################################

print('=============================================')
print("start export")
print('=============================================')

#define storage path for model:
model_path = path + "SpeechMLP/pretrained_MLP/pretrained_MLP/"
model_name_new = "testprobs"+str(buffernum2+1)+"_"+ model_name
print("Model saved to: ")
print(model_path + model_name_new)

#pickle:----------------------------------------------------------------

#~ #create export lists:
#~ testhistory = [test_filenames_history, test_softmax_list]

#~ #export trained model:
#~ f = open(model_path + model_name, "wb")
#~ pickle.dump(model_name, f,protocol=2)
#~ pickle.dump(testhistory, f,protocol=2)
#~ f.close()

#.mat file:-------------------------------------------------------------

#new emission probs:
modelw = {"model_name" : model_name}
modelw["test_filenames_history"] = test_filenames_history
modelw["test_softmax_list"] = test_softmax_list

#from training:
#modelweights
modelw["W1"] = modelweights[0]
modelw["W2"] = modelweights[1]
modelw["W3"] = modelweights[2]
modelw["W4"] = modelweights[3]
modelw["W5"] = modelweights[4]
modelw["W6"] = modelweights[5]
modelw["b1"] = modelweights[6]
modelw["b2"] = modelweights[7]
modelw["b3"] = modelweights[8]
modelw["b4"] = modelweights[9]
modelw["b5"] = modelweights[10]
modelw["b6"] = modelweights[11]
#opt: modellayout
modelw["nodes"] = modellayout[0]
modelw["classnum"] = modellayout[1]
modelw["framenum"] = modellayout[2]
modelw["acttype"] = modellayout[3]
#opt: trainingparams
modelw["randomint"] = trainingparams[0]
modelw["epochs"] = trainingparams[1]
modelw["learnrate"] = trainingparams[2]
modelw["batsize_train"] = trainingparams[3]
modelw["batsize_test"] = trainingparams[4]
#opt: history = [cost_history, train_acc_history, test_acc_history]
modelw["cost_history"] = history[0]
modelw["train_acc_history"] = history[1]
modelw["test_acc_history"] = history[2]
#phoneme transcriptions
modelw["phoneme_labels"] = phoneme_labels
modelw["label_transcript"] = label_transcript


scipy.io.savemat(model_path + model_name_new,modelw)

print('=============================================')
print("export finished")
print('=============================================')
print(" "+"\n")


########################################################################
#plot settings:
########################################################################

#=======================================================================
#plot testing
#=======================================================================

#~ #plot test accuracy function
#~ plt.figure(1, figsize=(8,6))
#~ plt.plot(range(len(test_acc_history)),test_acc_history, color='m')
#~ plt.axis([0,len(test_acc_history),0,1])
#~ plt.title('Test Accuracy')
#~ plt.xlabel('number of batches, batch size: '+ str(batsize_test))
#~ plt.ylabel('accuracy percentage')

#~ #plt.ion()
#~ plt.show()
#~ #plt.hold(False)
