
#!/usr/bin/env python
#coding: utf-8
'''
toy example for 5 state HMM
decode german number: Eins -> ains
'''
import numpy as np
import os
import csv
import scipy.io

#get path dir:
scriptdir = os.getcwd()
#os.chdir("../")
path = os.getcwd()

########################################################################
# init global paths
########################################################################

# choose data dirs
# eri
file_dir0 = '//hmm_eri//words_from_mono//'
#file_dir0 = '//hmm_eri//words_from_mono_no_sil//'
# rvg
file_dir = '//hmm_rvg_digits-small_mono//words_from_mono//'
#file_dir = '//hmm_rvg_digits-small_mono//words_from_mono_no_sil//'
# hmm filelist
data_name = file_dir0
path_final = path+data_name

# path to word labels
path_w_label = path+'//w_label//'

########################################################################
# define functions
########################################################################

def load_w_labels(path, filename):
	'''load list of word labels for HMMs'''
	#~ w_label_list = os.listdir(path)
	#~ print('w_label_list')
	#~ print(w_label_list)
	get_w_label = []
	# read in file
	reader = csv.reader(open(path+filename), delimiter=' ')
	for line in reader:
		#print(line)
		#print(line[2])
		get_w_label.append(line[2][2:])
	#print(get_w_label)
	return get_w_label

def load_hmms(path_final):
	'''load list of hmm files (Transition Matrices A)'''
	hmm_file_list = os.listdir(path_final)
	print('hmm_file_list')
	print(hmm_file_list)
    #init A for all hmms
	hmm_A_list = [] 
	for hmm_file in hmm_file_list:
		# init params
		get_A = []
		pos = 0
		A_cols = np.empty(0)
		# read in file
		reader = csv.reader(open(path_final+hmm_file), delimiter=' ')
		for line in reader:
			if line[0] == '~h':
				# get transcription: word
				get_trans = line[1]
			if line[0] == '<NUMSTATES>':
				# get number of states
				num_states = int(line[1])
				AA = np.zeros([0,num_states])
			if line[0] == '<TRANSP>':
				# get transistion probs
				A_cols = np.arange(pos+1,pos+1+num_states)
			if pos in A_cols:
				# create transistion matrix A
				get_A = line
				get_A =[float(i) for i in get_A[1:num_states+1]]
				AA_newline = np.array(get_A)
				AA = np.vstack([AA, AA_newline])
			pos = pos + 1
		hmm_A_list.append(AA)
		#~ print("HMM transcription")
		#~ print(get_trans)
		#~ print("NUM STATES")
		#~ print(num_states)
		#~ print("transition probs A ")
		#~ print(AA)
	return hmm_file_list, hmm_A_list

def export_mat(hmm_file_list,hmm_A_list,mat_name):
	'''export: .mat file method'''
	#create dict model
	data = {"hmm_name" : data_name}
	data["hmm_filenames"] = hmm_file_list
	data["hmm_trans_A_list"] = hmm_A_list		
	scipy.io.savemat(path + mat_name,data)

def viterbi(A, B, A_label):
	'''Viterbi Decoder'''
	A_len = len(A)
	B_len = len(B)
	with np.errstate(divide='ignore'):
		A = np.log10(A)
		B = np.log10(B)
		#~ print('A')
		#~ print(A)
		#~ print('B')
		#~ print(B)
		#~ print(A_label)
		V = np.log10(np.zeros_like(B))
		# init backtracking
		Phi = np.log10(np.zeros_like(B))
		Phi2 = np.log10(np.zeros_like(B))
	########################################
	# Start Decoding
	########################################
	# init, time = 0
	V[A_len-2,0] = B[A_len-2,0]  #P2(0)
	# start recursion
	for time in range(1,B_len):
		# loop over time
		for state in range(1,A_len-1):
			# loop over states
			path_ij = A[state,state+1] + V[A_len-state,time-1]
			path_jj = A[state,state] + V[A_len-state-1,time-1]
			Max_val = max(path_ij, path_jj)
			Max_idx = np.argmax(np.array([path_ij, path_jj]))
		    # apply recursion equation
			V[A_len-state-1,time] = B[A_len-state-1,time] + Max_val
			# store backtracking
			Phi[A_len-state-1,time-1]  = Max_val
			Phi2[A_len-state-1,time-1]  = Max_idx
			
	# termination
	Log_prob = A[A_len-2,A_len-1] + V[1,B_len-1]
	# start backtracking
	Phi[0,B_len-1] = Log_prob;

	
	#~ print('===================')
	#~ print('B')
	#~ print(B.shape)
	#~ print(B)
	#~ print('V')
	#~ print(V.shape)
	#~ print(V)				
	#~ print('Phi_val')
	#~ print(Phi)
	#~ print('Phi2_location')
	#~ print(Phi2)
	#~ print('===========================')
	#~ print('decoded:  ',A_label)
	#~ print('Log_prob: ',Log_prob )
	#~ print('===========================')
	return A_label,Log_prob

def extract_phonem(hmm_file_list, hmm_A_list ,phonem):
	'''seperate sil and sp from HMM list'''
	idx_phonem = [i for i, s in enumerate(hmm_file_list) if phonem in s]
	# if phonem is in list extract and remove it
	if len(idx_phonem) != 0:	
		hmm_name_phonem = hmm_file_list[idx_phonem[0]]
		hmm_A_phonem = hmm_A_list[idx_phonem[0]]
		del hmm_file_list[idx_phonem[0]]
		del hmm_A_list[idx_phonem[0]]
		return hmm_file_list, hmm_A_list, hmm_name_phonem, hmm_A_phonem 
	else:
		return hmm_file_list, hmm_A_list, 0 , 0

def fake_softmax(A):
	'''Generate artifical NN softmax outputs'''
	Softmax = np.zeros_like(A)
	Softmax  = Softmax [0:len(Softmax )-2,0:len(Softmax)]
	for b_pos in range(len(A)):
		np.random.seed(b_pos)
		rng = np.random.rand(len(Softmax))
		rng_bn = rng /np.sum(rng)
		Softmax [:,b_pos] = rng_bn
		#print(b_pos)
		#print(Softmax)
		#~ #print(np.sum(Softmax))
	#print(Softmax)
	return Softmax


def series_hmm(w_label,hmm_file_list,hmm_A_list):
	'''creates HMMs piped into series'''
	print('w_label_series')
	print(w_label)
	wl_len = len(w_label)
	hmm_part_A_old = np.empty(shape=[0],dtype=float)
	hmm_concat = np.empty(shape=[0],dtype=float)
	for phonem in w_label:
		idx_phonem = [i for i, s in enumerate(hmm_file_list) if phonem in s]
		if idx_phonem == []:
			print('  ERROR, phoneme is not in list!')
			print('  Check your HMM file list.\n')
		hmm_part = hmm_file_list[idx_phonem[0]]
		hmm_part_A = hmm_A_list[idx_phonem[0]]
		
		if hmm_part_A_old.shape[0] != 0:
			h_old = hmm_part_A_old.shape[0]
			h_new = hmm_part_A.shape[0]
			hc_size = h_old + h_new - 2
			hmm_concat = np.zeros([hc_size,hc_size])
			# concat hmms in series
			h_o = hmm_part_A_old[0:h_old-1,:]
			h_n = hmm_part_A[1:h_new-1,:]
			hmm_concat[0:h_old-1,0:h_old] = h_o
			hmm_concat[h_old-1:h_old-2+h_new-1,h_old-2:h_old-2+h_new] = h_n
			hmm_part_A_old = hmm_concat	
		else:
			hmm_part_A_old = hmm_part_A		
	#~ print(hmm_concat)
	#~ print(hmm_concat.shape)
	return hmm_concat

def hmm_sil(hmm_file_list,hmm_A_list, phonem):
	'''creates HMM list with sil at start and end'''
	print('hmm_file_list')
	#print(hmm_file_list)
	rm_clean = [['sil','sil','sil'],['sil','sp','sil']]
	hmm_file_sil_list = []
	hmm_A_sil_list = []
	for pos in range(len(hmm_file_list)):
		hmm_part = hmm_file_list[pos]
		
		new_w_label = [phonem,hmm_part,phonem]
		print(new_w_label)
		
		hmm_concat = series_hmm(new_w_label,hmm_file_list,hmm_A_list)
		hmm_file_sil_list.append(new_w_label)
		hmm_A_sil_list.append(hmm_concat)
		
	for rm in rm_clean:
		idx_phonem = [i for i,x in enumerate(hmm_file_sil_list) if x == rm]
		if idx_phonem != []:
			del hmm_file_sil_list[idx_phonem[0]]
			del hmm_A_sil_list[idx_phonem[0]]
			print('idx')
			print(idx_phonem)
	print('hmm_file_list')
	print(hmm_file_sil_list)
	print(len(hmm_file_sil_list))
	print(len(hmm_A_sil_list))
	return hmm_file_sil_list, hmm_A_sil_list
	
	
def hmm_loop(hmm_file_list,hmm_A_list):
	'''loops over all HMMs in the List'''
	#~ print(hmm_file_list)
	
	# init global HMM params:
	decoded_probs = np.empty(shape=[0],dtype=float)
	#start HMM recognition loop
	for hmm_file in range(len(hmm_file_list)):
		# A = transition probs
		A = hmm_A_list[hmm_file]
		A_label = hmm_file_list[hmm_file]
		#~ print(A)
		#~ #print(A.shape)
		#~ #print(A_label)
		
		# B = emission probs
		# create fake softmax:
		Softmax = fake_softmax(A)
		
		# pad Softmax for first/end state
		B_pad = np.zeros([1,len(A)])
		B = np.concatenate((B_pad,Softmax,B_pad),axis=0)
						 
		#~ # dictionary
		#~ #dict_ains = ['_','ai','n','s','_']
			
		# start viterbi decoder
		A_label,Log_prob = viterbi(A, B, A_label)
		print('===========================')
		print('hmm no.:  ',hmm_file+1)
		print('decoded:  ',A_label)
		print('Log_prob: ',Log_prob )
		print('===========================')
		
		decoded_probs = np.append(decoded_probs , Log_prob)
	print('decoded_probs')
	print(decoded_probs)
	prob_max = np.amax(decoded_probs)
	prob_argmax = np.argmax(decoded_probs)
	print('===========================')
	print('hmm_file_argmax')
	recognized_hmm = hmm_file_list[prob_argmax]                 
	print(recognized_hmm)
	print(prob_max)   
	print('===========================')
	
	return recognized_hmm, prob_max	




########################################################################
# load hmms
########################################################################
# load all hmms
hmm_file_list, hmm_A_list = load_hmms(path_final)
# optional export:
mat_name = '//hmm_eri_word_mono_x'
#export_mat(hmm_file_list,hmm_A_list,mat_name)

########################################################################
# create HMMs in series 
########################################################################

# get word label
w_label_name = 'is100001_024.lab' #e.g. zwei
w_label_name = 't6l00001_001.lab'
#w_label_name = 'AAaaaA3.22.lab'

#monophone labels not working
#w_label_name = 'AAaaaA2.1.lab'
#w_label_name = 'st100001_019.lab'

w_label = load_w_labels(path_w_label, w_label_name)
print('w_label')
print(w_label)

# not really necessary!!
# create HMMs in series from label
#hmm_concat = series_hmm(w_label,hmm_file_list,hmm_A_list)


#creates hmm with sil at start and end
hmm_file_list, hmm_A_list = hmm_sil(hmm_file_list,hmm_A_list,'sil')


########################################################################
# not necessary anymore!!!
# seperate sil and sp from list
########################################################################
#~ ex_phon = ['sil','sp']
#~ hmm_ex_file = []
#~ hmm_ex_A = []
#~ for phon in ex_phon:
	#~ hmm_file_list, hmm_A_list, hmm_file_phon, hmm_A_phon = extract_phonem(hmm_file_list, hmm_A_list ,phon)
	#~ if hmm_file_phon != 0:
		#~ hmm_ex_file.append(hmm_file_phon)
		#~ hmm_ex_A.append(hmm_A_phon)

########################################################################
# start viterbi decoder
########################################################################
#recognize utterance

# insert softmax here in function: notice from NN turn off fake softmax !!
# insert label or pat info from NN here

recognized_hmm, prob_max = hmm_loop(hmm_file_list,hmm_A_list)


########################################################################
#define metric, wer, confusion matrix
########################################################################























