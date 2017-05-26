import re, os
import numpy as np

########################################################################
#init parameter:
########################################################################

#get path dir:
scriptdir = os.getcwd()
#os.chdir("../")
path = os.getcwd()
datapath = "/data/rvg_new/pattern_hghnr_39coef/"
labelpath = "/data/rvg_new/nn_output/"
categories =['is','phr','sp1','st1','std','t6','t7']
randomint = 2
trainpart = 0.8
valpart = 0.1
headerbytes = 12

########################################################################
#define functions:
########################################################################

def getFilelist(filepath):
	filelist = []
	for index, filename in enumerate(sorted(os.listdir(filepath))):
		filelist.append(filename)
		#print '{0:02d}. {1}'.format(index + 1, filename)
	return filelist
	

def sortCategories(categories, filelist):
	groupedcats=[]
	for cat in range(len(categories)):
		#init empty category:
		category=[]	
		for pos in range(len(filelist)):
			#pattern matching at the beginning of a string:
			matchObj = re.match(categories[cat], filelist[pos])
			#print(matchObj)
			if matchObj != None:
				category.append(filelist[pos])
		groupedcats.append(category)
	return groupedcats


def createRandomvec(randomint, datalength):
	np.random.seed(randomint)
	randomvec = np.arange(datalength)
	np.random.shuffle(randomvec)
	return randomvec

	
def splitData(categories):
	datavec = []
	randvec = []
	randvec_train = []
	randvec_val = []
	randvec_test = []	
	for cat in range(len(categories)):
		groupnum = len(categories[cat])
		trainnum = int(len(categories[cat]) * trainpart)
		valnum = int(len(categories[cat]) * valpart)
		testnum = groupnum - (trainnum + valnum)
		datalength = [groupnum,trainnum,valnum,testnum]	
		datavec.append(datalength)
		
		#create random Lists:
		randlist = createRandomvec(randomint, groupnum)
		randvec.append(randlist)
		#split up random vectors:
		randlist_train = randlist[0:trainnum]
		randlist_val = randlist[trainnum:trainnum+valnum]
		randlist_test = randlist[trainnum+valnum:groupnum]
		randvec_train.append(randlist_train)
		randvec_val.append(randlist_val)
		randvec_test.append(randlist_test)
		
	datainfo = [datavec, randvec]	
	return datainfo, randvec_train, randvec_val, randvec_test	


def createSets(cat_data, randomlist):
	dataset = []
	for catpos in range(len(cat_data)):
		for numpos in range(len(randomlist[catpos])):
			dataobj = cat_data[catpos][randomlist[catpos][numpos]]
			dataset.append(dataobj)	
	return dataset

	
def loadPatData(filename):
	with open(inputfilename, 'rb') as fid:
		frames = np.fromfile(fid, dtype=np.int32) #get frames
		#print (frames[0])
		fid.seek(headerbytes, os.SEEK_SET)  # read in without header offset
		datafile = np.fromfile(fid, dtype=np.float32).reshape((frames[0], 39)).T  # read the data into numpy
	return datafile
	
		
########################################################################
#read in sorted names: data and labels
########################################################################	

datalist = getFilelist(datapath)
labellist = getFilelist(labelpath)
#~ print(len(datalist))
#~ print(datalist[0],datalist[50000])
#~ print(labellist[0],labellist[-1])

########################################################################
#split lists into categories:
########################################################################

grouped_data = sortCategories(categories, datalist)
grouped_label = sortCategories(categories, labellist)
#print(grouped_data[0][0:10])
#print(grouped_labels[0])

#***********************************************************************
#split data into training, validation and test set
#
#training: 	 80% 
#validation: 10%
#test:       10%
#
#random sampling from each category respectively
#***********************************************************************

datainfo, randvec_train, randvec_val, randvec_test = splitData(grouped_data)

trainingset_pat = createSets(grouped_data, randvec_train)
validationset_pat = createSets(grouped_data, randvec_val)
testset_pat = createSets(grouped_data, randvec_test)


#optional: export splitted dataset-names to disk:
#~ np.savetxt('trainingset.txt', trainingset_pat, delimiter=" ", fmt="%s")
#~ np.savetxt('validationset.txt', validationset_pat, delimiter=" ", fmt="%s")
#~ np.savetxt('testset.txt', testset_pat, delimiter=" ", fmt="%s")

########################################################################
#load files: data
########################################################################

inputfilename = datapath + grouped_data[0][10]
#~ print(inputfilename)

inputfile = loadPatData(inputfilename)
#~ print (inputfile)
#~ print(inputfile.shape)

########################################################################
#load files: labels
########################################################################

inputlabel = labelpath + grouped_label[0][0]
#print(inputlabel)
labelfile = np.loadtxt(inputlabel)
#print(labelfile)
#print(len(labelfile))





