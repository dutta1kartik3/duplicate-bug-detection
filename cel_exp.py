OA
from gensim.modeOAls import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import logging
from sys import argv
from gensim.models import Word2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree

from gensim import utils, matutils
from sklearn.metrics.pairwise import cosine_similarity as cs
from sklearn.metrics import log_loss as loss
from sklearn.metrics import roc_auc_score as auc
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import warnings
warnings.filterwarnings("ignore")


EPOCHS = 21
#MODEL_DIR = "/lustre/amar/Word2VecModels"
MODEL_DIR = "/lustre/amar/office_models"
def infer_bug_vector(words, dimensions, model, algorithm):
	if algorithm in ['pvdm', 'dbow']:
		return model.infer_vector(words)
	result = array([0.0]*int(dimensions))
	count = 0
	for word in words:
		if word in model:
			result+=model[word]
			count+=1
	return result/count


def val(algorithm, dimensions, test_filename, epoch):
	test_file = file(test_filename)
	for line in test_file:
		break #metaline
	model_filename = MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_%s.word2vec"%epoch
	model = Word2Vec.load(model_filename)
	slave_vectors = []
	master_vectors = []
	total_vectors = []
	actual_labels = []
	pred_labels = []
	c = 0
	s = 0
	m = 0
	for line in test_file:
	 	c+=1
		#try:
		line = line.strip().split("\t") 
			
			#slave = matutils.unitvec(array(infer_bug_vector(line[1].split(" ") + line[2].split(" "), int(dimensions)))).astype(REAL)
			#master = matutils.unitvec(array(infer_bug_vector(line[5].split(" ") + line[6].split(" "), int(dimensions)))).astype(REAL)
		slave_vec = list(infer_bug_vector(line[1].split(" ") +   line[2].split(" "), int(dimensions), model, algorithm))
		master_vec = list(infer_bug_vector(line[3].split(" ") + line[4].split(" "), int(dimensions), model, algorithm))

		if (np.isnan(slave_vec).any()):
			s+=1
		elif(np.isnan(master_vec).any()):
			m+=1
		else:
			label = abs(cs(slave_vec, master_vec)[0][0])
			actual_labels.append(int(line[5]))
			pred_labels.append(label)
		#print label
			#except ValueError:
			#	print "Val E"
		#except AttributeError:
		#	print "Att Error"
	print c,s,m
	print len(actual_labels), len(pred_labels)
	result = auc(actual_labels, pred_labels)
	result_file = file("OfficeValAucResults_"+ algorithm +".txt","a+")
	
	result = algorithm + "\t" +str(dimensions) + "\t" + str(epoch)+ "\t" + str(result)+"\n"

	result_file.write(result)
	result_file.close()
	test_file.close()

	print s, m



def __main__():
	algorithm = argv[1]
	dimensions = [50, 100, 150, 200, 250]
	test_file = argv[2]
	for d in dimensions:
		for i in xrange(1, EPOCHS, 1):
			val(algorithm, str(d), test_file, str(i))
__main__()	
