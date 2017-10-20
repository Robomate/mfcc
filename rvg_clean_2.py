#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
#=======================================================================
#Purpose: Clean RVG data 
#         change zwo - tsv'o:
#=======================================================================
'''
import numpy as np
import os
import re
import csv

########################################################################
#init parameter:
########################################################################

# init paths
path = "/home/student/Desktop/projekt_cleanrvg/rvg_clean"
#par_path  = '/data/rvg_new/update/v3.6_zi/PAR_LQ/'
par_path  = '/data/rvg_new/update/v3.6_test/PAR_LQ/'

########################################################################
# define functions:
########################################################################


def load_txt_files(file_path):
	'''load dataset-names from .txt files:'''
	#load dataset-names from .txt files:
	return np.genfromtxt(file_path, delimiter=" ",dtype='str')

def load_csv_files(file_path):
	'''load list .txt/.csv table'''
	changes = []
	# read in file
	f = open(file_path)
	reader = csv.reader(f, delimiter='\t')
	for line in reader:
		print(line)
		line = replace_exp(line, "tsv'aI", "tsv'o:")
		line = replace_exp(line, "2", "zwo")
		print(line)
		changes.append(line)
	f.close()
	
	
	####!!!!!!!!!!!!!!!!!!!!################do write file delimiter tab
	
	
	# write changes to file
	with open(file_path, 'ab') as f:
		writer = csv.writer(f)
		writer.writerows(changes)
		
		 
	print('changes')
	for change in changes:
		print(change)

		#outputFile = open(file_path, 'w', newline='')
		#outputWriter = csv.writer(outputFile)
		#outputWriter = csv.writer(reader)
		#csvWriter.writerow(['apples', 'oranges', 'grapes'])				

def replace_exp(line, old_exp, new_exp):
	'''replace expressions in lists'''
	matching = [s for s in line if old_exp in s]
	idx_matching = [i for i, s in enumerate(line) if old_exp in s]
	if idx_matching	== [2]:
		#print(matching)
		print(idx_matching)
		line[idx_matching[0]] = new_exp
		print(line)
	return line	
	
	


########################################################################
# init main:
########################################################################

# load filenames
file_list = load_txt_files(path)
fl = len(file_list)
print(fl)

# load par file:
file_par_1 = file_list[4][29:-8]+".par"
file_par_2 = file_par_1[:-9]
file_par_3 = file_par_1[8:]
file_par_4 = file_par_2 + '0' + file_par_3


filepath_par = par_path + file_par_4 
print(filepath_par)

# load par files
#~ file_par = load_txt_files(filepath_par)
#~ print(file_par)



file_par = load_csv_files(filepath_par)

#np.genfromtxt(file_par, delimiter=" ",dtype='str')





















