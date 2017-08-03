#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re, os
import numpy as np

#get path dir:
scriptdir = os.getcwd()
#os.chdir("../")
path2 = os.getcwd()

# init paths:
path 	 = "/home/korf/Desktop/BA_05/" #on korf@alien2 ,korf@lynx5(labor)
#path 	 = "/home/praktiku/korf/BA_05/" #on praktiku@dell9
#path 	 = "/home/praktiku/Videos/BA_05/" #on praktiku@dell8 (labor)
pathname = "00_data/RVG/dataset_filenames/"

#ERI corpus:------------------------------------------------------------
data_dir_eri = "00_data/ERI/pattern_hghnr_39coef/"
label_dir_eri = "00_data/ERI/nn_output_mono/"

datapath = path+data_dir_eri

########################################################################
#define functions:
########################################################################

def getFilelist(filepath):
	filelist = []
	for index, filename in enumerate(sorted(os.listdir(filepath))):
		filelist.append(filename)
		#print '{0:02d}. {1}'.format(index + 1, filename)
	print(len(filelist),'files loaded')
	return filelist
	


def save_to_disk(filelist):
	np.savetxt('eri_testset.txt', filelist, delimiter=" ", fmt="%s")

datalist = getFilelist(datapath)
print(datalist[0:10])
save_to_disk(datalist)
