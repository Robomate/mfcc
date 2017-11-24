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
#Inputs:  RVG new (German)
#		  training utterances: 		56127
#		  validation utterances: 	 7012
#         test utterances: 		 	 7023
#         --------------------------------
#		  total					    70162						
#
#		  shape of input: 1x273 vector (39MFCCcoeff * 7frames)
#
#Output:  mono: 135 classes (45 monophones with 3 HMM states each)
#		  tri: 4643 classes (tied triphones states out of 40014)
#
#Version: 6/2017 Roboball (MattK.)

#Start tensorboard via bash: 	tensorboard --logdir /logfile/directory
#Open Browser for tensorboard:  localhost:6006
#tensorboard --logdir /home/praktiku/korf/speechdata/tboard_logs/MLP_5layer_2017-05-10_14:04
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
import pandas as pd
import csv
from sklearn.metrics import confusion_matrix

import viterbi_loop_14 as viterbi

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

def loadMatfile(datapath):
	'''load pretrained model from .mat file:'''
	mat = scipy.io.loadmat(datapath)
	#get sorted list of dict keys:
	#print('mat_keys:')
	for key, value in sorted(mat.items()):
		pass
		#print (key)
	#print('===========================================')
	return mat

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
	#~ print('un padded datafile')
	#~ print(datafile)
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
	#~ print('padded datafile')
	#~ print(paddeddata.shape)	
	#~ print(paddeddata)		
	return paddedlabel, paddeddata
	
def slidingWin(framenum, paddedlabel, paddeddata,randomint):
	'''create sliding windows to extract chunks'''
	#init: empty lists
	labellist=[]
	datalist=[]
	if randomint != -11:
		# randomize frame order in training
		rnd_pos = createRandomvec(randomint, paddedlabel.shape[0]-framenum+1)
	#sliding window over utterance
	for pos in range(paddedlabel.shape[0]-framenum+1):
		if randomint != -11:
			pos = rnd_pos[pos]
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

def Preprocess(framenum, randbatnum, dataset_name, path,randomint,directory):
	'''load data and label files, pad utterance, create chunks'''
	#init params:		 
	headerbytes = 12
	#~ datapath  = path + data_dir
	#~ labelpath = path + label_dir
	datapath  = path + directory[0]
	labelpath = path + directory[1]
	#load data file:
	dataset_name_dat = dataset_name[randbatnum][:-4]+".pat"
	datafilename = datapath + dataset_name_dat
	datafile = loadPatData(datafilename, headerbytes)	
	#load label file:
	dataset_namelab = dataset_name[randbatnum][:-4]+".txt"
	inputlabel = labelpath + dataset_namelab
	labelfile = np.loadtxt(inputlabel) #read in as float
	#pad utterance:
	paddedlabel, paddeddata = padUtterance(framenum, labelfile, datafile)	
	#extract vector and labels by sliding window:
	labellist, datavectorlist = slidingWin(framenum, paddedlabel, paddeddata,randomint)	
	return labellist, datavectorlist

def Databuffer(buffersamples, framenum, dataset_name, path,randomint,directory):
	'''create buffer for minibatch'''
	#init buffer
	labelbuffer = []
	datavectorbuffer = []	
	for pos in range(buffersamples):		
		labellist, datavectorlist = Preprocess(framenum, pos, dataset_name, path,randomint,directory)
		labelbuffer = labelbuffer+labellist
		datavectorbuffer = datavectorbuffer+datavectorlist
	return labelbuffer, datavectorbuffer

def rand_buffer(label_buffer, data_buffer):
	'''randomize buffer order for minibatch'''
	SEED = 10
	random.seed(SEED)
	random.shuffle(label_buffer)
	random.seed(SEED)
	random.shuffle(data_buffer)
	return label_buffer, data_buffer
	
def phonemeDict():
	'''dictionary for classes'''
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

def load_lex(path, file_name):
	'''load lexicon with keys as words and vals as phonemes'''
	reader = csv.reader(open(path+file_name), delimiter=' ')
	word_dict = {line[0]: line[1:] for line in reader}
	return word_dict

########################################################################
# init parameter
########################################################################

print('=============================================')
print("load pretrained model")
print('=============================================\n')

#load pretrained model:
#mat = loadMatfile('pretrained_MLP/2017-07-17_11:53_MLP_4lay_17fs_257coeffs_512nodes_4643class_20eps_relu_bnorm_entry_no_lnorm_0.377testacc.mat')

#pretrain: RVG -> ERI 
mat = loadMatfile('pretrained_DNN_HMM/MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_relu_2017-06-15_19:46_0.601.mat') #wer 1.76 auf digits
#pretrain: ERI -> RVG
#mat = loadMatfile('pretrained_DNN_HMM/MLP_5lay_11fs_39coeffs_1024nodes_135class_20eps_relu_2017-06-15_19:46_0.601.mat') #wer 0.07 auf eri

# fix initial randomness:
randomint = 1
modtype = "MLP_pretrained_mixed_RVG_ERI" 
modlayer = 5	 	

# get timestamp:
timestamp = str(datetime.datetime.now())
daytime = timestamp[11:-10]
date = timestamp[0:-16]
timemarker = date+"_" + daytime

# init paths:
path 	 = "/home/student/Desktop/BA_05/" #on korf@alien2 ,korf@lynx5(labor)
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
#path 	 = "/home/praktiku/Videos/BA_05/" #on praktiku@dell8 (labor)
pathname = "00_data/dataset_filenames/"

# RVG corpus:-----------------------------------------------------------
data_dir_rvg = "00_data/RVG/pattern_hghnr_39coef/"
#data_dir = "00_data/RVG/pattern_dft_16k/"
#data_dir = "00_data/RVG/pattern_dft_16k_norm/"

label_dir_rvg = "00_data/RVG/nn_output/"
#label_dir = "00_data/RVG/nn_output_tri/"
#label_dir = "00_data/RVG/nn_output_tri_align1/"

directory_rvg = [data_dir_rvg,label_dir_rvg]

# ERI corpus:-----------------------------------------------------------
data_dir_eri = "00_data/ERI/pattern_hghnr_39coef/"
label_dir_eri = "00_data/ERI/nn_output_mono/"
directory_eri = [data_dir_eri,label_dir_eri ]


logdir = "tboard_logs/"
tboard_path = path + logdir
tboard_name = modtype + "_"+ str(modlayer)+ "layer_"+ str(timemarker)

# init filenames:
trainset_filename = 'trainingset.txt'
validationset_filename = 'validationset.txt'
testset_filename = 'testset.txt'
test_eri = 'eri_testset.txt'

#load filenames:
trainset_name = loadNamefiles(path + pathname + trainset_filename)
valset_name = loadNamefiles(path + pathname + validationset_filename)
testset_name = loadNamefiles(path + pathname + testset_filename)
test_eri_name = loadNamefiles(path + pathname + test_eri)
			
# init model parameter:
coeffs =  39                              #39, 257  
nodes = 2*512 							#size of hidden layer
classnum = 135							#classes mono: 135, tri:4643
framenum = 11							#number of frames
inputnum = framenum * coeffs 				#length of input vector
display_step = 100 						#to show progress
bnorm = 'bnorm_entry'
lnorm = 'no_lnorm'
optimizing = 'adam'
classifier = 'softmax'

# init activation function:
actdict = ["tanh", "relu", "selu","sigmoid"]
acttype = actdict[1]

# error message frame number:
if framenum%2==0:
	print("Error, please choose uneven number of frames.")
	
# init training parameter:
epochs = 20                       
learnrate = 1e-4                        #high:1e-4
#train_size  = len(trainset_name) 
train_size  = len(test_eri_name) 

val_size = len(valset_name)
tolerance = 0.01                        #break condition  
batsize_train = 256													
batches_train = int(train_size / batsize_train) #round down
buffer_train = 10 		#choose number utterances in buffer
train_samples = int(train_size/buffer_train)

# init test parameter:
#test_size = len(testset_name)
test_size = len(test_eri_name)	
	
batsize_test = 100
batches_test = int(test_size / batsize_test)   #round down
buffer_test = 10
test_samples = int(test_size/buffer_test)    #number of test_samples

# init emission probs parameter	
batsize_test2=1
buffer_test2=1
test_samples2=2

# random shuffle filenames:
valset_rand = randomShuffleData(randomint, valset_name)
# random shuffle weights inits:
rand_w = createRandomvec(randomint, modlayer+1)

# preform unit testing
epochs = 1
train_samples = 10
test_samples = 1

########################################################################
# init and define model:
########################################################################

# init model:
def weight_variable(pretrained_weight):
	'''init weights'''
	#initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(pretrained_weight, name="W")
	
def bias_variable(pretrained_bias):
	'''init biases'''
	#initial = tf.constant(0.1, shape=shape)
	return tf.Variable(pretrained_bias, name="b")

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

# init placeholder:
x_input = tf.placeholder(tf.float32, shape=[None, inputnum],name="input")
y_target = tf.placeholder(tf.float32, shape=[None, classnum],name="labels")

# init weights:	
# 1. hidden layer
with tf.name_scope(name="fc1"):
	W1 = weight_variable(mat[sorted(list(mat.keys()))[0]])
	b1 = bias_variable(mat[sorted(list(mat.keys()))[14]])
	tf.summary.histogram("weights", W1)
	tf.summary.histogram("biases", b1) 
# 2. hidden layer 
with tf.name_scope(name="fc2"):
	W2 = weight_variable(mat[sorted(list(mat.keys()))[1]])
	b2 = bias_variable(mat[sorted(list(mat.keys()))[15]])
	tf.summary.histogram("weights", W2)
	tf.summary.histogram("biases", b2)  
# 3. hidden layer
with tf.name_scope(name="fc3"):
	W3 = weight_variable(mat[sorted(list(mat.keys()))[2]])
	b3 = bias_variable(mat[sorted(list(mat.keys()))[16]])
	tf.summary.histogram("weights", W3)
	tf.summary.histogram("biases", b3) 
# 4. hidden layer
with tf.name_scope(name="fc4"):
	W4 = weight_variable(mat[sorted(list(mat.keys()))[3]])
	b4 = bias_variable(mat[sorted(list(mat.keys()))[17]])
	tf.summary.histogram("weights", W4)
	tf.summary.histogram("biases", b4)  
# 5. hidden layer
with tf.name_scope(name="fc5"):
	W5 = weight_variable(mat[sorted(list(mat.keys()))[4]])
	b5 = bias_variable(mat[sorted(list(mat.keys()))[18]])
	tf.summary.histogram("weights", W5)
	tf.summary.histogram("biases", b5)
# 6. output layer
with tf.name_scope(name="fc6"):
	W6 = weight_variable(mat[sorted(list(mat.keys()))[5]])
	b6 = bias_variable(mat[sorted(list(mat.keys()))[19]])
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
print("model: "+str(modlayer)+ " hidden layer "+str(modtype))
print("**********************************")
print("hidden units: "+str(nodes)+" each layer")
print("activation function: "+str(acttype))
print("optimizer: Adam")
print("----------------------------------")
print("data name: RVG new")
print("training data: " +str(train_size))
print("validation data: " +str(val_size))
print("test data: " +str(test_size)+"\n")


	
########################################################################
#start testing:
########################################################################

def testing(test_samples,batsize_test,buffer_test, randomint_test,directory):
	'''test the neural model'''
	print('=============================================')
	print('start '+str(modtype)+' testing')
	print('=============================================')
	t2_1 = datetime.datetime.now()
	
	# random shuffle filenames:	
	testset_rand = randomShuffleData(randomint_test, testset_name)
	if directory[0] == "00_data/ERI/pattern_hghnr_39coef/":
			testset_rand = randomShuffleData(randomint_test, test_eri_name)
			
	#insert fix input for debugging 
	#print(testset_rand[0:10])
	testset_rand = ['AAaagA4.9.txt','AAaagA4.92.txt','AAaagA4.91.txt'] #from commands_8khz_waves.file
	#print(testset_rand[0:10])
	
	# init histories
	test_acc_history = np.empty(shape=[0],dtype=float)
	test_filenames_history = np.empty(shape=[0],dtype=str)
	test_softmax_list = []
	label_transcript_list = []
	test_max_transcript_list = []
	random_int = -11
	
	####################################################################
	# init dictionaries
	####################################################################
	# get phonem dictionaries
	p_dict_mono_45 = phonemeDict()
	#p_dict_mono_45
	if test_samples < 20:
		pass
		#~ print("Phonem Dictionary: ")
		#~ print(p_dict_mono_45)
		#~ print(" ")
	# get word dictionaries
	#~ w_dict_rvg_digit_10 = load_lex(path+'00_data/lexicon/','lex_rvg_digit_10.csv')
	#~ w_dict_eri_64 = load_lex(path+'00_data/lexicon/', 'lex_eri_64.csv')
	
	
	####################################################################
	#              init anal_recog params ( 3 structs)
	#
	############## init 3 structs: ref , dis , syn #####################
	####################################################################

	# struct 1. ref (= hmmfile = refs)
	########################################################
	# init ref parameter: ref.struct with  12 fields
	# alternative to above -> load eri_word.mat and
	# save in ref (= hmmfile = refs)!!!!
	# so all hmms parameter are in ref
	########################################################
	#load word.mat----------------------------------------------
	hmm_mat = loadMatfile('hmm_eri/eri_word.mat')
	#hmm_mat = loadMatfile('hmm_rvg_digits-small_mono/rvg_word.mat')
	########################################################
	# ref.vectorsize  = 39 (MFCC)
	ref_vectorsize  = hmm_mat['refw']['vectorsize'][0][0][0][0]
	# ref.no_of_refs = 65 (words, hmms)        # used in viterbi
	ref_no_of_refs = hmm_mat['refw']['no_of_refs'][0][0][0][0]
	#~ print('ref_no_of_refs')
	#~ print(ref_no_of_refs)
	# ref.numstates           --fehlt!!!!!
	# ref.nummixes            -- not needed
	# ref.means               -- not needed
	# ref.variances           -- not needed
	# ref.weight_of_mixtures  -- not needed
	# ref.transp ( transprobs A)
	ref_transp = []
	for xx in range(len(hmm_mat['refw']['transp'][0][0][0])):
		hmm_x = hmm_mat['refw']['transp'][0][0][0][xx]
		ref_transp.append(hmm_x)
	#~ print(len(hmm_mat['refw']['transp'][0][0][0]))
	#~ print(hmm_mat['refw']['transp'][0][0][0][9].shape)
	#~ print(hmm_mat['refw']['transp'][0][0][0][9])
	#~ print('ref_transp[0]')
	#~ print(len(ref_transp))
	#~ print(ref_transp[0].shape)
	#~ print(ref_transp[64].shape)	
	#~ print(ref_transp[64])		
	#hmm_list = [hmm_mat['refw']['transp'][0][0][0][-1],hmm_mat['refw']['transp']]
	#~ hmm_list = [hmm_l[-1],ref_transp]
	#~ print('hmm_list[0]')
	#~ print(hmm_list[1][0].shape)
	#~ print(hmm_list[1][0])
	# ref.name         --fehlt!!!!!
	#~ print(hmm_mat['refw']['name'][0][0][0][9])
	# ref.silstr
	# ref.type
	# ref.phonindex         word list     # used in viterbi
	# copy for local prob
	#~ local_prop = np.copy(hmm_mat['refw']['phonindex'])
	#local_prop = local_prop.astype(np.float32)	
	#~ print(hmm_mat['refw']['phonindex'][0][0][0].shape)
	#~ print(hmm_mat['refw']['phonindex'][0][0][0][9][0].shape)
	#~ print(hmm_mat['refw']['phonindex'][0][0][0][9][0])
	########################################################

	# struct 2: syn
	########################################################
	# init syn parameter: syn.struct with  11 fields
	# checked with matlab: yes
	########################################################
	# syn.num_endnodes
	syn_num_endnodes = 1
	# syn.num_of_nodes
	syn_num_of_nodes = 2
	# syn.name_of_refs (sil +65 woerter)   --fehlt
	# syn.entry_nodes
	syne = np.ones(65)
	syne[64] = 2
	syn_entry_nodes = [np.ones(1),syne]
	#print(syn_entry_nodes)
	# syn.no_of_refs
	syn_no_of_refs = [1,65]
	# syn.endnodes
	syn_endnodes = 2
	# syn.num_of_synonyms
	syn_num_of_synonyms = 64
	# syn.syn_refs   (64 , 3woerter hintereinander) ---fehlt
	# syn.num_of_synrefs
	syn_num_of_synrefs = 2 * np.ones(64)
	# syn.timeout
	syn_timeout = 80
	# syn.ref_index
	syn_ref_index = [np.array([64]),np.arange(0,65)]
	#print('ref',len(np.arange(1,66)))
	########################################################	

	# struct 3. dis
	########################################################
	# init dis parameter: dis.struct with  11 fields
	########################################################
	# dis.best_ref
	dis_best_ref = np.zeros(2)       
	# dis.best_from
	dis_best_from = np.zeros(2)
	# dis.sumprob
	dis_sumprob_l = []
	dis_sumprob = []
	# dis.oldsum
	dis_oldsum_l = []      #like sumprob
	dis_oldsum = []  #like sumprob
	# dis.from_frame
	dis_from_frame_l = [] #like sumprob
	dis_from_frame = [] #like sumprob
	# dis.oldframe
	dis_oldframe_l = []  #like sumprob
	dis_oldframe = []  #like sumprob
	# dis.node_prob
	#dis_node_prob = [np.log(np.zeros(1)),np.log(np.zeros(1))]
	dis_node_prob = [np.log(0),np.log(0)]
	# dis.local_prob
	dis_local_prob = []
	# dis.weight       (1x65 cell)     --fehlt !!
	# dis.minskip
	dis_min_skip = np.zeros(65)
	# dis.maxskip
	dis_max_skip = np.ones(65)
	########################################################
	
	
	#test loop: length = int(test_size/buffersamples),test_samples
	for buffernum in range(int(1)):
			
			#grab linear minibatch datavectors from databuffer:
			testset_buffernum = testset_rand[buffernum * buffer_test: (buffernum * buffer_test)+buffer_test]
			print('testset_buffernum') #['AAaagA4.9.txt']
			print(testset_buffernum)
			test_filenames_history = np.append(test_filenames_history,testset_buffernum)
			
			#create buffer:
			labelbuffer, datavectorbuffer = Databuffer(buffer_test, framenum, testset_buffernum, path,random_int,directory)	
			bufferlength = len(labelbuffer)
			#~ print("buffernum: "+str(buffernum))
			#~ print("bufferlength: "+str(bufferlength))
			
			#~ print('datavectorbuffer')
			#~ print(len(datavectorbuffer))
			#~ print(datavectorbuffer[0].shape)
			
			#get label to phoneme transcription:
			label_transcript = phonemeTranslator(labelbuffer, p_dict_mono_45)
			#get number of minibatches from buffer:
			minibatches = int(bufferlength/batsize_test)
			#~ print("number of minibatches: "+str(minibatches))		
			#init histories
			test_softmax_utterance = np.empty(shape=[0,classnum],dtype=float)	
			
			#minibatch loop per buffer: minibatches
			for batch in range(2):
				
				#init viterbi
				bestref = np.ones([syn_num_of_nodes,minibatches])
				fromeframe = np.ones([syn_num_of_nodes,minibatches])
				print('batch')
				print(batch)
				print('================')
				#grab minibatch datavectors from databuffer:
				datavector = datavectorbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]		
				#grab minibatch labels from labelbuffer:		
				labelvector = labelbuffer[batch * batsize_test: (batch * batsize_test)+batsize_test]
				#~ print('datavector')
				#~ print(len(datavector))
				#~ print(datavector[0].shape)
							
				#minibatch data:
				testbatchdata = createMinibatchData(batsize_test, framenum, datavector)
				#minibatch labels: one hot encoding
				testbatchlabel = createMinibatchLabel(batsize_test, classnum, labelvector)
				#~ print('testbatchdata')
				#~ print(testbatchdata.shape)
				
				#start feeding data into the model:
				feedtest = {x_input: testbatchdata, y_target: testbatchlabel}
				predictions = accuracy.eval(feed_dict = feedtest)
				test_acc_history = np.append(test_acc_history,predictions)
				
				#if test_samples < 20:
				#get emissionprobs
				softmaxprobs = sess.run(softmax,feed_dict=feedtest)
				test_softmax_utterance = np.append(test_softmax_utterance,softmaxprobs, axis=0)
				#~ print('test_softmax_utterance')
				#~ print(test_softmax_utterance.shape)
				#~ print(test_softmax_utterance)
				
				########################################################
				# init 3 structs: ref (htk info), dis , syn
				# plus 3 other parameter
				#
				######## ref is from htk (with hmm info) ###############
				# first load parameter from eri:
				# from matlab eri-wordmat
				#
				# ref (in viterbi use ref) = hmmfile = refw
				# ref (struct in matlab)
				# re.vectorsize  = 39 (MFCC)
				# ref.no_of_refs = 65 (words)        # used in viterbi
				# ref.numstates
				# ref.nummixes
				# ref.means
				# ref.variances
				# ref.weight_of_mixtures
				# ref.transp
				# ref.name
				# ref.silstr
				# ref.type
				# ref.phonindex                      # used in viterbi
				#
			    ######## syn ###########################################
				# syntax file
				#
				# syn.num_endnodes
				# syn.num_of_nodes
				# syn.name_of_refs
				# syn.entry_nodes
				# syn.no_of_refs
				# syn.endnodes
				# syn.num_of_synonyms
				# syn.syn_refs
				# syn.num_of_synrefs
				# syn.timeout
				# syn.ref_index
				#
				######## dis ###########################################
				# dis in matlab file for viterbi init
				# in matlab           | in python
				#
				# dis.best_ref        | 
				# dis.best_from
				# dis.sumprob
				# dis.oldsum
				# dis.from_frame
				# dis.oldframe
				# dis.node_prob
				# dis.local_prob
				# dis.weight
				# dis.minskip
				# dis.maxskip
				#
				######## others ########################################
				# other parameter in matlab file for viterbi init
				#
				# bestref
				# nf
				# fromframe
				#
				
				########################################################
				
				########################################################
				# finish init parameter: from dis (fill lists)
				########################################################
				# assign softmax values to hmm labels
				for pos in range(ref_no_of_refs):
					#print('==========')
					#print(pos)
					lab_hmm = hmm_mat['refw']['phonindex'][0][0][0][pos][0]
					lprob = np.zeros_like(lab_hmm).astype(np.float32)
					# init sumprob, oldsum, from_frame, oldframe
					if batch == 0:
						sumprob = np.zeros_like(lab_hmm).astype(np.float32)
						sumprobx = np.append(sumprob,0)
						sumproby = np.log(sumprobx) 
						dis_sumprob_l.append(sumproby)	
						dis_oldsum_l.append(sumproby)
						dis_from_frame_l.append(sumprobx)
						dis_oldframe_l.append(sumprobx)
							
					#print('==========')
					#do for all hmms:
					for pos2 in range(len(lab_hmm)):
						lprob[pos2] = np.log(max(1e-6,softmaxprobs[0][lab_hmm[pos2]-1]))
						#~ print(pos2)
						#~ print(lprob[pos2])
					#print(lprob)
					dis_local_prob.append(lprob)  # in matlab: code 577
				
				# finish init dis: complete list in list
				if batch == 0:
					dis_sumprob = [[dis_sumprob_l[-1]],dis_sumprob_l] #init with inifinity
					dis_oldsum = [[dis_oldsum_l[-1]],dis_oldsum_l]  #init with inifinity
					dis_from_frame = [[dis_from_frame_l[-1]],dis_from_frame_l] #init with zeros
					dis_oldframe = [[dis_oldframe_l[-1]],dis_oldframe_l] #init with zeros
					
				#-------------------------------------------------------
				# optional: print different init lists	
				#-------------------------------------------------------	
				#~ print(lprob_list[64])
				#~ print('dis_sumprob')
				#~ print(dis_sumprob_l[64])
				#~ print(dis_sumprob_l[-1])	
				#~ print('softmaxprobs')	
				#~ print(softmaxprobs.shape)
				#~ print(softmaxprobs[0][3])
				#~ print(softmaxprobs[0][4])	
				#~ print(hmm_mat['refw']['transp'][0][0][0][9])
				#-------------------------------------------------------
				
				count = batch #till 193
				########################################################
				# ab hier: start viterbi decoding 
				# for one softmax output: (1,135) 
				# till no_of_frames = 1:193
				########################################################
				print()
				print('=========================================')
				print('------------------------')
				print('start viterbi decoding')
				print('------------------------')
				print('=========================================')
				print()
				# loop over all hmms
				for node_ind in range(syn_num_of_nodes):  #loop times: 2
					#print(syn_num_of_nodes)
					#~ print('node_ind',node_ind)
					#~ print('==========')
					# loop over each model
					for ref_ind in range (syn_no_of_refs[node_ind]):  #loop times: 1
						#print(syn_no_of_refs[node_ind])
						#print(ref_ind)
						#print(ref_ind)
						#print(int(syn_entry_nodes[node_ind][ref_ind]))
						
						# start with state 1-----------------------------
						#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        # spring beim ersten count=1 hierhin von continue
                        # von: 614 -704 , if ( count == 1 ) to %look at all nodes of the syntax
                        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
						if count == 0 and int(syn_entry_nodes[node_ind][ref_ind]) == 1: 
							for pos5 in range(1,int(dis_max_skip[int(syn_ref_index[node_ind][ref_ind])])+1):
								#~ print("kuku",node_ind)
								#~ print("kuku",ref_ind)
								#~ print("kuku",pos5)
								#~ print(int(syn_ref_index[node_ind][ref_ind]))
	                            # evtl letztes hmm noch einfügen zaehlt nur bis 64!!!
								#dis_sumprob[node_ind][ref_ind][pos5] = dis_local_prob[int(syn_ref_index[node_ind][ref_ind])][pos5-1] + hmm_mat['refw']['transp'][0][0][0][ref_ind][0,pos5]
								dis_sumprob[node_ind][ref_ind][pos5] = dis_local_prob[int(syn_ref_index[node_ind][ref_ind])][pos5-1] + ref_transp[int(syn_ref_index[node_ind][ref_ind])][0,pos5]
								#~ print(hmm_mat['refw']['transp'][0][0][0][ref_ind][0,pos5])  #fehlt noch
								#~ print(dis_sumprob[node_ind][ref_ind])
								#~ print(dis_sumprob[node_ind][ref_ind].shape)	
								dis_from_frame[node_ind][ref_ind][pos5] = -1
								#print(dis_from_frame[1][62])	
								
						# skip the rest of the loop for the first node	
						if count == 0 :
							continue
						# end with state 1------------------------------
						
						#evtl ab hier noch mal kontrollieren!!!!
						
						
						# calculation of accumulated probability for all states > 1
						# assign new parameter
						# matlab line 620 - 653
						# wird erst fuer count=1 interessant im ersten loop
						print('count(batch)',count)
						print('node_ind',node_ind)
						print('ref_ind',ref_ind)
						oldframe = dis_oldframe[node_ind][ref_ind] #letzte eintrag stimmt evtl. nicht 
						print('oldframe')
						print(dis_oldframe[1][63])
						print(oldframe)
						print(oldframe.shape)
						local_prob = dis_local_prob[int(syn_ref_index[node_ind][ref_ind])] #erster eintrag stimmt nicht 
						#~ print('dis_local_prob[int(syn_ref_index[node_ind][ref_ind])]')
						#~ print(count)
						#~ print(dis_local_prob[int(syn_ref_index[node_ind][ref_ind])].shape)
						#~ print(dis_local_prob[int(syn_ref_index[node_ind][ref_ind])])
						oldsum = dis_oldsum[node_ind][ref_ind]
						#~ print('oldsum')
						#~ print(oldsum.shape)
						#~ print(oldsum)
						#transmission probs
						transp = ref_transp[int(syn_ref_index[node_ind][ref_ind])]
						#~ print('transp')
						#~ print(transp.shape)
						#~ print(transp)
						#~ print('hmm_list[0]')
						#~ print(hmm_list[0].shape)
						#~ print(hmm_list[0])
						max_skip = dis_max_skip
						min_skip = dis_min_skip
						num_states	=  hmm_mat['refw']['numstates'][0][0][0]
						
						###################################################################
						
						for state_ind in range(1,num_states[int(syn_ref_index[node_ind][ref_ind])]+1):
							#print(num_states[int(syn_ref_index[node_ind][ref_ind])]+1)
							############################################
							# start function max_search_opt
							############################################
							refliste_ind = int(syn_ref_index[node_ind][ref_ind])
							#~ print('refliste_ind')
							#~ print(refliste_ind)
							#j = state_ind
							minskip_x = min_skip[refliste_ind]
							maxskip_x = max_skip[refliste_ind]
							dmax = oldsum[state_ind] + transp[state_ind][state_ind]
							#~ print('dmax')
							#~ print(dmax) 
							#~ print(oldsum[state_ind]) 
							#~ print(transp[state_ind][state_ind]) 
							max_ind = 0
							j = state_ind + 1
							#~ print('j')
							#~ print(j)
							#~ print('minskip_x')
							#~ print(minskip_x)
							if minskip_x > 0:
								pass
								#print(num_states[refliste_ind]+1)
								while j <= num_states[refliste_ind]+1 and minskip_x > 0:
									#print(num_states[refliste_ind]+1)
									
										
									#ab hier noch unchecked"!!!!!!!!!!!!
									
									#use np.amax for index!!!
									max(dmax,dis_oldsum[node_ind][ref_ind][j]+ ref_transp[refliste_ind][j,state_ind])  #fehlt noch index und location!!!!!!!!!
									if ind == 2:
										max_ind = state_ind - j
									j = j + 1
									minskip_x = minskip_x -1
							
							j = state_ind - 1
							#print(j)
							while j >= 0 and maxskip_x > 0:
								pass
								new = oldsum[j] + transp[j][state_ind]
								#~ print(oldsum[j])
								#~ print(transp[j][state_ind])
								#print(new)
								if new > dmax:
									max_ind = state_ind - j
									dmax = new
								j = j-1
								maxskip_x = maxskip_x -1	
							############################################
							# end function max_search_opt
							############################################
							#~ print('node_ind',node_ind)
							#~ print('ref_ind',ref_ind)
							#~ print('stateind',state_ind)			
							dis_from_frame[node_ind][ref_ind][state_ind] = oldframe[state_ind - max_ind]	
							#~ print(oldframe[state_ind - max_ind])
							#~ print(dis_from_frame[node_ind][ref_ind])
							dis_sumprob[node_ind][ref_ind][state_ind] =  local_prob[state_ind -1] + dmax
							#~ print(local_prob[state_ind -1] + dmax)
							#~ print(dis_sumprob[node_ind][ref_ind])	
								
							
				#~ print(count)
				num_states = hmm_mat['refw']['numstates'][0][0][0]
				
				
				
				#ab hier nochmal kontorllierien!!!!!!!!!!!!!!!!!!!!!
				
				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                # spring beim ersten count=1 hierhin von continue
                # von: 614 -704 , if ( count == 1 ) to %look at all nodes of the syntax
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				
				
				############################################
				# start function calc_node_prob
			    ############################################
				for node_ind in range(syn_num_of_nodes):
					dis_node_prob[node_ind] = np.log(0)
					dis_best_ref[node_ind] = -1 
					dis_best_from[node_ind] = -1
					#~ print(dis_node_prob)
					#~ print(dis_best_ref)
					#~ print(node_ind)
					#~ print(syn_no_of_refs[node_ind]-1)
					for ref_ind in range(syn_no_of_refs[node_ind]):
						state_ind = num_states[int(syn_ref_index[node_ind][ref_ind])] + 1
						#~ print('state_ind')
						#~ print(state_ind)
						skip = dis_max_skip[int(syn_ref_index[node_ind][ref_ind])]
						print(skip)
						
						while state_ind-1 > 0 and skip > 0: #evtl ist skip noch falsch oder auch state_ind!!!!!!!!
							dum = dis_sumprob[node_ind][ref_ind][state_ind-1]
							print(dum)
							if dum > np.log(0):
								#!!!!!! noch falsch!!!!!!!!!!!!!
								dum = dum + ref_transp[int(syn_ref_index[node_ind][ref_ind])][state_ind-1,num_states[int(syn_ref_index[node_ind][ref_ind])] + 2]
								if dum > dis_node_prob[node_ind]:
									dis_node_prob[node_ind] = dum
									dis_best_ref[node_ind] = ref_ind
									#evtl noch falsch!!!!!!!!!!!!!
									dis_best_from[node_ind] = dis_from_frame[node_ind][ref_ind][state_ind-1]
									
							state_ind = state_ind -1
							skip = skip -1
				
				for node_ind in range(syn_num_of_nodes):
					#~ print('node_ind')
					#~ print(node_ind)
					
					for ref_ind in range(syn_no_of_refs[node_ind]):
						#noch flasch!!!!
						print(int(syn_entry_nodes[node_ind][ref_ind]))
						
						if dis_node_prob[int(syn_entry_nodes[node_ind][ref_ind])-1] > np.log(0):
							dis_sumprob[node_ind][ref_ind][0] = dis_node_prob[int(syn_entry_nodes[node_ind][ref_ind])-1]
							dis_from_frame[node_ind][ref_ind][0] = count
						else:
							dis_sumprob[node_ind][ref_ind][0] = np.log(0)
							dis_from_frame[node_ind][ref_ind][0] = 0
							
							
				############################################
				# end function calc_node_prob
				############################################
				
				
				for node_ind in range(syn_num_of_nodes):
					for ref_ind in range(syn_no_of_refs[node_ind]):
						dis_oldsum[node_ind][ref_ind] = dis_sumprob[node_ind][ref_ind]
						dis_oldframe[node_ind][ref_ind] = dis_from_frame[node_ind][ref_ind]
						
				
				############################################
				# end  function viterbi sync
				############################################
				print(dis_best_ref.shape)
				#~ dis_best_ref = np.expand_dims(dis_best_ref, axis=0).T
				#~ dis_best_from = np.expand_dims(dis_best_from, axis=0).T
				print(dis_best_ref)
				
				bestref[:,count]  = dis_best_ref
				fromeframe[:,count] = dis_best_from
				#~ print(bestref)
				#~ print('fromeframe')
				#~ print(fromeframe)
				#~ print(fromeframe.shape)
				#~ print(count)
				
				
				############################################
				# end of loop for one file
				############################################
				print('Total probability: ', max(dis_node_prob))
				
				
				
				
				
				
				
				
				
				############################################
				# start backtrack viterbi sync 
				# (for recognizing the sequence , the word)
				############################################
				
				
				############################################
				# end of backtrack viterbi sync 
				############################################
				
				#~ call(r)
										
						
						
						
						
						
						
						
						
						# end with all states > 1-----------------------
						
					
							
					
			############################################################
			# old : start viterbi decoder
			############################################################
				
			# turn off fake softmax in viterbi !!
			#~ print('test_softmax_utterance.shape')
			#~ print(test_softmax_utterance.shape)
			#~ #print(test_softmax_utterance [0:10,0:10])
			#~ print(test_softmax_utterance)
			#print(np.log(test_softmax_utterance [0:10,0:10]))
			
			# load word labels:
			#w_label = viterbi.load_w_label(path_w_label, testset_buffernum[0])
			#~ print(w_label)
			#~ print(hmm_file_list_sil)
			#~ print(hmm_A_list_sil[0].shape)
			#print(np.log(hmm_A_list_sil[0][0:7,0:7]))
			
			
			#~ #load word.mat----------------------------------------------
			#~ hmm_mat = loadMatfile('hmm_eri/eri_word.mat')
			#~ #hmm_mat = loadMatfile('hmm_rvg_digits-small_mono/rvg_word.mat')
			
			#~ # print out word list
			#~ print(hmm_mat['refw']['name'][0][0][0][9])
			#~ # print out A
			#~ print(hmm_mat['refw']['transp'][0][0][0][9].shape)
			#~ print(hmm_mat['refw']['transp'][0][0][0][9])
			
			#~ # print out labels of A
			#~ print(hmm_mat['refw']['phonindex'][0][0][0].shape)
			#~ print(hmm_mat['refw']['phonindex'][0][0][0][9].shape)
			#~ print(hmm_mat['refw']['phonindex'][0][0][0][9])
			#~ # print number of hmms
			#~ ref_no_of_refs = hmm_mat['refw']['no_of_refs'][0][0][0][0]
			#~ print(ref_no_of_refs)	
			
			#--------------------------#################################
			
	
			
		    # recognize utterance
			#recognized_hmm, prob_max = viterbi.hmm_loop(hmm_file_list_sil,hmm_A_list_sil, test_softmax_utterance,w_label,p_dict_mono_45,w_dict_eri_64)
			
			############################################################
			# compare w_label with recognized hmm
			############################################################
			
			#~ print('w_label')
			#~ print(w_label)
			#~ print('recognized_hmm')
			#~ print(recognized_hmm)
			
			#~ # remove any 'sil'
			#~ label_result = [x for x in w_label if x != 'sil']
			#~ rec_result = [x for x in recognized_hmm if x != 'sil']
			#~ print('w_label')
			#~ print(label_result)
			#~ print('recognized_hmm')
			#~ print(rec_result)
			
			#~ # lists for confusion matrix
			#~ label_list = label_list + label_result
			#~ pred_list  = pred_list + rec_result
			#~ print('label_list')
			#~ print(label_list)
			#~ print('pred_list')
			#~ print(pred_list)
			
			#~ # compare w_label with recognized hmm
			#~ if rec_result == label_result:
				#~ correct = correct + 1
						
			
			############################################################
			#print out softmax testing info
			############################################################
			#~ if test_samples < 20:
				#~ #save emission-probs for each utterance
				#~ test_softmax_list.append(test_softmax_utterance)
				
				#~ #translation to phonemes
				#~ testmax_loc = test_softmax_utterance.argmax(axis=1)
				#~ test_max_transcript = np.zeros_like(testmax_loc).astype(str)
				#~ for loc in range(len(testmax_loc)):
					#~ test_max_transcript[loc] = p_dict_mono_45[testmax_loc[loc]]
				#~ test_max_transcript_list.append(test_max_transcript)
				#~ label_transcript_list.append(label_transcript)
				#~ print('=============================================')
				#~ print("test utterance: " + str(buffernum+1))
				#~ print("label name: " + testset_buffernum[0]+"\n")
				#~ print("label_transcript: ")
				#~ print(str(label_transcript))
				#~ print('test_max_transcript: ')
				#~ print(str(test_max_transcript)+"\n")
		
			#check progress: test accuracy
			if  buffernum%10 == 0:
				test_accuracy = accuracy.eval(feed_dict=feedtest)			
				t2_2 = datetime.datetime.now()
				print('test utterance: '+ str(buffernum*buffer_test)+'/'+str(test_size))
				print('test accuracy: %.2f'% test_accuracy + 
				" -- test time: " + str(t2_2-t2_1)[0:-7])
	
	####################################################################
	#viterbi: metric, wer, confusion matrix
	####################################################################
	
	#~ # print info: lists for confusion matrix
	#~ print('label_list')
	#~ print(label_list)
	#~ print('pred_list')
	#~ print(pred_list)
		
	#~ # WER
	#~ print('================================')
	#~ print('WER')
	#~ print('================================')	
	#~ wer = 1 - (correct/len(pred_list))
	#~ print(wer)
	
	#~ print('================================')
	#~ print('confusion_matrix')
	#~ print('================================')
	#~ # print labels of confusion matrix
	#~ print(hmm_file_list)
	#~ conf_mat = confusion_matrix(label_result, rec_result,labels=hmm_file_list)
	#~ print(conf_mat)

	####################################################################
	#print out neural network testing info
	####################################################################			
	print('=============================================')
	# total test stats:
	print('test utterance: '+ str(test_size)+'/'+str(test_size))
	total_testacc = np.mean(test_acc_history, axis=0)
	print("overall test accuracy %.3f"%total_testacc)       
	t2_3 = datetime.datetime.now()
	test_time = t2_3-t2_1
	print("total test time: " + str(test_time)[0:-7]+'\n')		
	
	#~ # info to start tensorboard:
	#~ print('=============================================')
	#~ print("tensorboard")
	#~ print('=============================================')
	#~ print("to start tensorboard via bash enter:")
	#~ print("tensorboard --logdir " + tboard_path + tboard_name)
	#~ print("open your Browser with:\nlocalhost:6006")
	
	testparams_list = [test_acc_history, test_filenames_history, 
					   test_softmax_list, randomint_test]
	
	return testparams_list, total_testacc, test_time, p_dict_mono_45

########################################################################
#start main: 
########################################################################

########################################################################
#start testing: 
########################################################################

# RVG traininig testing-------------------------------------------------
#~ #start training: 
#~ train_acc_history, cost_history, crossval_history, train_time, total_trainacc = training(epochs,train_samples,directory)
#~ # start testing: full set
#~ testparams_list, total_testacc, test_time, phoneme_labels  = testing(test_samples,batsize_test,buffer_test,1,directory)
#~ # start testing: emission probs
#~ testparams_list2, total_testacc2, test_time2, phoneme_labels = testing(test_samples2,batsize_test2,buffer_test2,2,directory_rvg)

# ERI traininig testing-------------------------------------------------
# start training eri: 
#train_acc_history, cost_history, crossval_history, train_time, total_trainacc = training(epochs,train_samples,directory_eri)

# start testing: full set
#testparams_list, total_testacc, test_time, phoneme_labels  = testing(test_samples,batsize_test,buffer_test,1,directory_eri)
# start testing: emission probs
testparams_list2, total_testacc, test_time2, phoneme_labels = testing(test_samples2,batsize_test2,buffer_test2,2,directory_eri)

########################################################################
#plot settings:
########################################################################

#~ #plot model summary:
#~ plot_model_sum = "model_summary: "+str(modtype)+"_"+str(modlayer)+" hid_layer, "+\
                          #~ " hid_units: "+str(nodes)+" ,"+" frames: "+\
                          #~ str(framenum)+", classes: " +str(classnum)+", epochs: "+str(epochs)+"\n"+\
                          #~ bnorm+ ", "+ lnorm+", "+"training accuracy %.3f%total_trainacc"+\
                          #~ " , test accuracy %.3f"%total_testacc+" , "+\
                          #~ str(timemarker)
                          
#~ #define storage path for model:
#~ model_path = path + "07_DNN_HMM/pretrained_DNN_HMM/"
#~ model_name = str(timemarker)+"_"+modtype + "_"+ str(modlayer)+ "lay_"+ str(framenum)+ "fs_"+\
             #~ str(coeffs)+ "coeffs_"+ str(nodes)+ "nodes_"+\
             #~ str(classnum)+ "class_" +str(epochs)+"eps_"+\
             #~ acttype+"_"+ bnorm+ "_"+ lnorm+\
             #~ "_"+str(total_testacc)[0:-9]+"testacc"+modtype
#~ print("Model saved to: ")
#~ print(model_path + model_name)
                  
#init moving average:
#~ wintrain = 300
#~ wintest = 300
#~ wtrain_len = len(cost_history)
#~ if wtrain_len < 100000:
	#~ wintrain = 5
#~ wtest_len = len(testparams_list[0])
#~ if wtest_len < 10000:
	#~ wintest = 5
	
#=======================================================================
#plot training
#=======================================================================

#~ #plot training loss function
#~ fig1 = plt.figure(1, figsize=(8,8))
#~ plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
#~ plt.subplots_adjust(wspace=0.5,hspace=0.5)
#~ plt.subplot(211)
#~ plt.plot(range(len(cost_history)),cost_history, color='#1E90FF')
#~ plt.plot(np.convolve(cost_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#97FFFF', alpha=1)
#~ plt.axis([0,len(cost_history),0,int(10)+1])
#~ #plt.axis([0,len(cost_history),0,int(np.max(cost_history))+1])
#~ plt.title('Cross Entropy Training Loss')
#~ plt.xlabel('number of batches, batch size: '+ str(batsize_train))
#~ plt.ylabel('loss')

#~ #plot training accuracy function
#~ plt.subplot(212)
#~ plt.plot(range(len(train_acc_history)),train_acc_history, color='#20B2AA')
#~ plt.plot(np.convolve(train_acc_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#7CFF95', alpha=1)
#~ plt.axis([0,len(cost_history),0,1])
#~ plt.title('Training Accuracy')
#~ plt.xlabel('number of batches, batch size:'+ str(batsize_train))
#~ plt.ylabel('accuracy percentage')

#~ #export figure
#~ plt.savefig('imagetemp/'+model_name+"_fig1"+'.jpeg', bbox_inches='tight')
#~ im1 = pltim.imread('imagetemp/'+model_name+"_fig1"+'.jpeg')
#~ #pltim.imsave("imagetemp/out.png", im1)

#=======================================================================
#plot testing
#=======================================================================

#~ #plot validation loss function
#~ plt.figure(2, figsize=(8,8))
#~ plt.figtext(.5,.95,plot_model_sum, fontsize=10, ha='center')
#~ plt.subplots_adjust(wspace=0.5,hspace=0.5)
#~ plt.subplot(211)
#~ plt.plot(range(len(crossval_history)),crossval_history, color='#1E90FF')
#~ plt.plot(np.convolve(crossval_history, np.ones((wintrain,))/wintrain, mode='valid'), color='#87CEFA', alpha=1)
#~ plt.axis([0,len(crossval_history),0,int(10)+1])
#~ #plt.axis([0,len(crossval_history),0,int(np.max(crossval_history))+1])
#~ plt.title('Cross Validation Loss')
#~ plt.xlabel('number of validation checks')
#~ plt.ylabel('loss')

#~ #plot test accuracy function
#~ plt.subplot(212)
#~ plt.plot(range(len(testparams_list[0])),testparams_list[0], color='m')
#~ plt.plot(np.convolve(testparams_list[0], np.ones((wintest,))/wintest, mode='valid'), color='#F0D5E2', alpha=1)
#~ plt.axis([0,len(testparams_list[0]),0,1])
#~ plt.title('Test Accuracy')
#~ plt.xlabel('number of batches, batch size: '+ str(batsize_test))
#~ plt.ylabel('accuracy percentage')

#~ #export figure
#~ plt.savefig('imagetemp/'+model_name+"_fig2"+'.jpeg', bbox_inches='tight')
#~ im2 = pltim.imread('imagetemp/'+model_name+"_fig2"+'.jpeg')

########################################################################
#start export:
########################################################################

#~ print('=============================================')
#~ print("start export")
#~ print('=============================================')

#~ print("Model saved to: ")
#~ print(model_path + model_name)

#~ W1 = np.array(sess.run(W1))
#~ W2 = np.array(sess.run(W2))
#~ W3 = np.array(sess.run(W3))
#~ W4 = np.array(sess.run(W4))
#~ W5 = np.array(sess.run(W5))
#~ W6 = np.array(sess.run(W6))
 
#~ b1 = np.array(sess.run(b1))
#~ b2 = np.array(sess.run(b2))
#~ b3 = np.array(sess.run(b3))
#~ b4 = np.array(sess.run(b4))
#~ b5 = np.array(sess.run(b5))
#~ b6 = np.array(sess.run(b6))

#~ #.mat file method:------------------------------------------------------

#~ #create dict model
#~ model = {"model_name" : model_name}
#~ #modelweights
#~ model["W1"] = W1
#~ model["W2"] = W2
#~ model["W3"] = W3
#~ model["W4"] = W4
#~ model["W5"] = W5
#~ model["W6"] = W6
#~ model["b1"] = b1
#~ model["b2"] = b2
#~ model["b3"] = b3
#~ model["b4"] = b4
#~ model["b5"] = b5
#~ model["b6"] = b6
#~ model["acttype1"] = acttype
#~ model["acttype2"] = acttype
#~ model["acttype3"] = acttype
#~ model["acttype4"] = acttype
#~ model["acttype5"] = acttype
#~ #modellayout
#~ model["hiddenlayer"] = modlayer
#~ model["nodes"] = nodes
#~ model["classnum"] = classnum
#~ model["framenum"] = framenum
#~ model["randomint"] = randomint
#~ model["classification"] = classifier 
#~ model["optimizer"] = optimizing
#~ #trainingparams
#~ model["epochs"] = epochs
#~ model["learnrate"] = learnrate
#~ model["batsize_train"] = batsize_train
#~ model["total_trainacc"] = total_trainacc
#~ #testparams
#~ model["batsize_test"] = batsize_test
#~ model["randomint_test"] = testparams_list[3]
#~ model["total_testacc"] = total_testacc
#~ #history = [cost_history, train_acc_history, test_acc_history]
#~ model["cost_history"] = cost_history
#~ model["train_acc_history"] = train_acc_history
#~ model["test_acc_history"] = testparams_list[0]

#~ #from testing: files and emission probs
#~ model["test_filenames_history"] = testparams_list2[1]
#~ model["test_softmax_list"] = testparams_list2[2]
#~ model["batch_norm"] = bnorm
#~ model["layer_norm"] = lnorm
#~ model["phoneme_labels"] = phoneme_labels
#~ #modelw["label_transcript_list"] = label_transcript_list
#~ #modelw["test_max_transcript_list"] = test_max_transcript_list

#~ #save plot figures
#~ model["fig_trainplot"] = im1
#~ model["fig_testplot"] = im2

#~ total_trainacc = 0

#~ #export to .csv file
#~ model_csv = [timemarker,total_trainacc,total_testacc,
             #~ modtype,modlayer,framenum, coeffs, nodes,classnum,
             #~ classifier,epochs,acttype,learnrate,optimizing,
             #~ bnorm,lnorm,batsize_train,batsize_test, 'No',modtype]
#~ df = pd.DataFrame(columns=model_csv)

#~ #save only good models
#~ if total_trainacc > 0.25:
	#~ scipy.io.savemat(model_path + model_name,model)
	#~ df.to_csv('nn_model_statistics/nn_model_statistics.csv', mode='a')
	#~ df.to_csv('nn_model_statistics/nn_model_statistics_backup.csv', mode='a')
	
#~ print('=============================================')
#~ print("export finished")
#~ print('=============================================')
#~ print(" "+"\n")

#~ #print out model summary:
#~ print('=============================================')
#~ print("model summary")
#~ print('=============================================')
#~ print("*******************************************")
#~ print("model: "+ str(modtype)+"_"+ str(modlayer)+" hidden layer")
#~ print("*******************************************")
#~ print("hidden units: "+str(nodes)+" each layer")
#~ print("frame inputs: "+str(framenum))
#~ print("activation function: "+str(acttype))
#~ print("optimizer: Adam")
#~ print("-------------------------------------------")
#~ print("data name: RVG new")
#~ print("training data: " +str(train_size))
#~ print("validation data: " +str(val_size))
#~ print("test data: " +str(test_size))
#~ print("-------------------------------------------")
#~ print(str(modtype)+' training:')
#~ print("total training time: " + str(train_time)[0:-7])
#~ print("overall training accuracy %.3f"%total_trainacc ) 
#~ print("-------------------------------------------")
#~ print(str(modtype)+' testing:')
#~ print("total test time: " + str(test_time2)[0:-7])	
#~ print("overall test accuracy %.3f"%total_testacc) 
#~ print("*******************************************")


#plot show options:----------------------------------------------------------
#plt.show()
#plt.hold(False)
#plt.close()




