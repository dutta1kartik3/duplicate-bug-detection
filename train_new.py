from sys import argv
from gensim.models import Word2Vec
from random import shuffle

import logging
import itertools

import numpy as np
import gensim
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


dimensions = int(argv[1])
num_epochs = int(argv[2])
algo = int(argv[3])
print dimensions, num_epochs, algo
train_file = file("Train_file_Word2Vec.txt")
sentences = []
print "Reading Training File"
for line in train_file:
	line = line.strip().replace("\t"," ")
	for i in line:
		sentences.append(i.split(" "))
print "Read Training File"
model = Word2Vec(sg=algo, size=dimensions, negative=5,  min_count=1, workers=100)

model.build_vocab(sentences)
print "vocab built"
for epoch in range(0,num_epochs+1):	
	shuffle(sentences)	
	model.train(sentences)	
	model.save("/lustre/amar/tokenized_Word2Vec_models/Word2Vec_model_dimensions_"+str(dimensions)+"algo_"+ str(algo) +"_epoch_"+str(epoch)+".word2vec")
	

    

