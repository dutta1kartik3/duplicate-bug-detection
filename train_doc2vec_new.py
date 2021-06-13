from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
#import gensim.models.doc2vec.TaggedDocument as TG
from gensim.models.doc2vec import TaggedDocument as TG
EPOCHS = 51
MODEL_DIR = "/lustre/amar/tokenized_Word2Vec_models"
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_model(dimensions, name):

	pvdm = Doc2Vec(dm=1, size=dimensions, negative=5, min_count=2, workers=100)
	dbow = Doc2Vec(dm=0, size=dimensions, negative=5, min_count=2, workers=100, dbow_words = 1)
    
	if (name == 'pvdm'):
		return pvdm
	if (name == 'dbow'):
		return dbow


time_file = open("TimeAnalysis.csv","a+")

def train_model(model, sentences, dimensions, algorithm):
	epoch = 1
	t = time.time()
	model.build_vocab(sentences)
	while(epoch < EPOCHS):
		shuffle(sentences)	
		model.train(sentences, total_examples=model.corpus_count, epochs =1 )	
		if epoch%1 == 0:
			model.save(MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec")
		epoch += 1
	s = algorithm + ","+str(dimensions)+","+ str(EPOCHS) +"," + str(time.time() - t) + "\n"
	
	time_file.write(unicode(s))

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
	    doc = TG(words = doc, tags = ['tag'])		
	  
	    sentences.append(doc)
	return sentences
def __main__():
	algorithms = [ 'pvdm', 'dbow'] #'glove', 'varembed', 'wordrank']
	algorithm = argv[1]
	if (algorithm not in algorithms):
		print "Algorithm not available in Doc2Vec:", algorithm
		return
	dimensions = int(argv[2])
	model = get_model(dimensions, algorithm)
	
	senteces = get_sent('TrainFile.csv')
	 	
	train_model(model, senteces, dimensions, algorithm)
	time_file.close()



__main__()
