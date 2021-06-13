from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
from random import shuffle
from gensim.models.doc2vec import TaggedDocument


import logging
import itertools

import numpy as np
import gensim
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

data = {0:"Both",1:"Titles", 2:"Desc"}

dimensions = int(argv[1])
num_epochs = int(argv[2])
algo = int(argv[3])
source = int(argv[4])
ns = 5

train_file = file("Train_Cleansed.txt")
sentences = []
count = 0
for line in train_file:
	doc = ""
	line = line.strip().split("\t")
	if source == 0:
		doc = " ".join(line).split()
	elif source == 1:
		doc = line[0].split()
	elif source == 2:
		if len(line)>1:
			doc = line[1].split()
	else:
		#doc = 123
		print "Exit"
	if doc!="":	
		sentences.append(doc)
	count+=1


model =	Word2Vec(sg=algo, size=dimensions, negative=ns,  min_count=2, workers=50)

model.build_vocab(sentences)
for epoch in range(0,num_epochs+1):	
	shuffle(sentences)	
	model.train(sentences)	
	model.save("/lustre/amar/Word2Vec_models/"+ str(data[source]) +	"_Word2Vec_model_algo"+str(algo)+"__dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+"_ng_"+str(ns)+".word2vec")
					
