#!/usr/bin/env python
#coding: utf-8
'''
single word lexicon
'''
import numpy as np
import os
import csv

#get path dir:
scriptdir = os.getcwd()
#os.chdir("../")
path = os.getcwd()


def load_lex(path, file_name):
	'''load lexicon with keys as words and vals as phonemes'''
	reader = csv.reader(open(path+file_name), delimiter=' ')
	word_dict = {line[0]: [line[1:]] for line in reader}
	return word_dict

# get word dictionaries
w_dict_rvg_digit_10 = load_lex(path+'//', 'lex_rvg_digit_10.csv')
w_dict_eri_64 = load_lex(path+'//', 'lex_eri_64.csv')


print('w_dict_rvg_digit_10')
print(w_dict_rvg_digit_10)

print('w_dict_eri_64')
print(w_dict_eri_64)
