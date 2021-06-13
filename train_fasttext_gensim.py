from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
from gensim.models.word2vec import LineSentence as LS
import fasttext
EPOCHS = 51
#MODEL_DIR = "/lustre/amar/Word2VecModels"
MODEL_DIR = "/lustre/amar/office_models"
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



time_file = open("TimeAnalysis.csv","a+")



def get_sent(filename):
	sentences = []
	f = file(filename)

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
            sentences.append(doc)
	return sentences
def __main__():
	algorithms = ['skipgram', 'cbow'] #'glove', 'varembed', 'wordrank']
	algorithm = argv[1]
	if (algorithm not in algorithms):
		print "Algorithm not available:", algorithm
		return
	dimensions = int(argv[2])
	epochs = int(argv[3])
	traning_file = argv[4]
	print 'starting training...'
	output = MODEL_DIR + "/fast" + algorithm + "_model_dimensions_" + str(dimensions) + "_epoch_" + str(epochs)+".word2vec"

 	t = time.time()	
	if algorithm == 'skipgram':
		model = fasttext.skipgram(argv[4], output = output, dim = dimensions, epoch = epochs, min_count = 2, maxn = 8, thread = 50)		
	elif algorithm == 'cbow':
		 model = fasttext.cbow(argv[4], output = output, dim = dimensions, epoch = epochs, min_count = 2, maxn = 8, thread = 50)
	time_file.write(unicode("fast"+algorithm + ","+str(dimensions)+","+ str(epochs) +"," + str(time.time() - t) + "\n"))
	time_file.close()
	print  'fast', algorithm, epochs, 'done'


__main__()
