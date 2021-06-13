from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
from gensim.models.word2vec import LineSentence as LS
from gensim.models.wrappers.fasttext import FastText



def get_sent(filename):
	sentences = []
	f = file(filename)
	out_file = open('OfficeWrappersTrainFile.txt','w+')
	for line in f:
	    line.split('\t')
	    break
		    
	for line in f:
	    line = line.strip().split('\t')
	    if len(line)<3:
	    	continue
	    doc = line[2]
	    if len(line) > 3 and not line[3].isdigit():
		doc += line[3] 
	    doc = doc.split()
	    doc = filter(lambda x: len(x) > 2, doc)
	    doc = " ".join(doc) + "\n"
	    out_file.write(unicode(doc))
	out_file.close()
	
def __main__():
	get_sent('OfficeTrainFile.csv')
 	
__main__()
