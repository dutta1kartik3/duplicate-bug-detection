from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
EPOCHS = 51
MODEL_DIR = "/lustre/amar/Word2VecModels"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def get_model(dimensions, name):

	skipgram = Word2Vec(sg=1, size=dimensions, negative=5, min_count=2, workers=100)
	cbow = Word2Vec(sg=0, size=dimensions, negative=5, min_count=2,workers=100)
	pvdm = Doc2Vec(dm=1, size=dimensions, negative=5, min_count=2, workers=50)
	dbow = Doc2Vec(dm=0, size=dimensions, negative=5, min_count=2, workers=50, dbow_words = 1)

	if (name == 'skipgram'):
		return skipgram
	if (name == 'cbow'):
		return cbow
	if (name == 'pvdm'):
		return pvdm
	if (name == 'dbow'):
		return dbow


time_file = open("TimeAnalysis.csv","a+")

def train_model(model, sentences, dimensions, algorithm):
	epoch = 0
	t = time.time()
	model.build_vocab(sentences)
	while(epoch < EPOCHS):
		shuffle(sentences)	
		model.train(sentences, total_examples=model.corpus_count)	
		if epoch%2 == 0:
			model.save(MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec")
		epoch += 1
	time_file.write(algorithm + ","+str(dimensions)+","+ str(EPOCHS) +"," + str(time.time() - t) + "\n")

def build_vocab_nondups(filename):
	f = open(filename, encoding = 'ascii', errors = 'ignore')
	senteces = []
	for line in f:
		print line
		break
	for line in f:
		temp = []
		line = line.split("\t")
		l = [2, 3]
		try:
			for i in l:
				temp += line[i].lower().split()
		except IndexError:
			#print line
			pass
		senteces.append(temp)
	return senteces

def build_vocab_dups(filename):
	f = open(filename, encoding = 'ascii', errors = 'ignore')
	senteces = []
	for line in f:
		print line
		break
	for line in f:
		temp = []
		line = line.split("\t")
		l = [2, 3 ,  7 ,8]
		try:
			for i in l:
				temp += line[i].lower().split()

		except IndexError:
			#print line
			pass
		senteces.append(temp)
	return senteces

def __main__():
	algorithms = ['skipgram', 'cbow', 'pvdm', 'dbow'] #'glove', 'varembed', 'wordrank']
	algorithm = argv[1]
	if (algorithm not in algorithms):
		print "Algorithm not available:", algorithm
		return
	dimensions = int(argv[2])
	non_dups = 'Non_Dups.csv' 
	dups = 'DupsTrain.csv'

	sentences_non_dups = build_vocab_nondups(non_dups)
	sentences_dups = build_vocab_dups(dups)
	model = get_model(dimensions, algorithm)
	
	senteces = sentences_dups + sentences_non_dups
	print 'Sentences: ', len(senteces) 	
	train_model(model, senteces, dimensions, algorithm)
	time_file.close()



__main__()
