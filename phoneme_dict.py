import numpy as np

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
	#print("labels utterance: "+str(labels)), delimiter=','
	#print(label_transcript)	
	return label_transcript
	
def loadData(filename):
	"""load data from binary .raw format"""
	datafile = np.loadtxt(filename, delimiter='\n',dtype=str,usecols=(0),skiprows=0)
	return datafile	
	
#filename = 'monophon_hghnr39.hmm'
#filename = 'triphon_list'
filename = 'tiedlist'
	
datafile = loadData(filename)	
print(datafile)
print(len(datafile))
print(datafile[0])
		
#get label-dictionary:
#~ phoneme_labels = phonemeDict()
#~ print(phoneme_labels)
