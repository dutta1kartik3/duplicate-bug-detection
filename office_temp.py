from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
EPOCHS = 51
MODEL_DIR = "/lustre/amar/office_models"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#ve(MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec")


algorithm = argv[1]
dimensions = argv[2]
epoch = argv[3]

model_filename = MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec"
if algorithm in ['pvdm', 'dbow']:
	model = Doc2Vec.load(model_filename)
else:
	model = Word2Vec.load(model_filename) 


words = ['bugs', 'bug', 'firefox', 'email', 'emails', 'word', 'excel','calc', 'office', 'microsoft', 'file', 'xml', 'crash', 'java', 'math','code','problem', 'linux', 'pdf', 'document', 'impress', 'font', 'size']
	

qfile = open('OfficeQualitativeAnalysis_'+algorithm+'.tsv', "w+")

for word in words:
	i = model.wv.most_similar(word, topn =10)
	qfile.write(unicode(word + "\t" + str(i)+"\n"))
qfile.close()
