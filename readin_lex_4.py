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
	word_dict = {line[0]: line[1:] for line in reader}
	return word_dict
	
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
	
def find_phon(lexicon, word):
	'''find phoneme sequenz in lexicon'''
	idx_list = []
	phones = lexicon.get(word)
	print(phones)
	for phon in phones:
		matching = np.where(mono3s_dict == phon)
		print(phon)
		print('matching')
		print(matching[0])
		idx_list.append(matching[0])
	return idx_list
		
	
	
mono3s_dict = phonemeDict()
print('mono3s_dict')
print(mono3s_dict)
	

# get word dictionaries
w_dict_rvg_digit_10 = load_lex(path+'//', 'lex_rvg_digit_10.csv')
w_dict_eri_64 = load_lex(path+'//', 'lex_eri_64.csv')


print('w_dict_rvg_digit_10')
print(w_dict_rvg_digit_10)

#~ print('w_dict_eri_64')
#~ print(w_dict_eri_64)

#print pretty:
#~ for keys in w_dict_rvg_digit_10:
    #~ print (keys)
    #~ for vals in w_dict_rvg_digit_10[keys]:
        #~ print (vals)

# search for word in dict
# RVG
word = 'drei'
idx_list = find_phon(w_dict_rvg_digit_10, word)
print(idx_list)


# ERI
wordx = 'ja'
idx_listx = find_phon(w_dict_eri_64, wordx)
print(idx_listx)









