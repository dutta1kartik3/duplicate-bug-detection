from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import time
from random import shuffle
from io import open
EPOCHS = 51
MODEL_DIR = "/lustre/amar/tokenized_Word2Vec_models"

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#ve(MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec")


algorithm = argv[1]
dimensions = argv[2]
epoch = argv[3]

model_filename = MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec"


if algorithm in ['pvdm', 'dbow']:
	model = Doc2Vec.load(model_filename)
elif algorithm in ['fastskipgram', 'fastcbow']:
	from gensim.models.wrappers import FastText
	model = FastText.load_fasttext_format(model_filename )
else:
	model = Word2Vec.load(model_filename) 


words = ['bugs', 'bug', 'firefox', 'email', 'emails', 'thunder', 'thunderbird', 'error', 'browser', 'windows', 'element', 'css', 'html', 'javascript', 'java', 'python', 'font'	,'crash', 'click', 'url','c++', 'cpp']
	

qfile = open('QualitativeAnalysis_'+algorithm+'.tsv', "w+")

for word in words:
	i = model.wv.most_similar(word, topn =10)
	qfile.write(unicode(word + "\t" + str(i)+"\n"))
qfile.close()
