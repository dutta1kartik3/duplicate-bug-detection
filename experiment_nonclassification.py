from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
import logging
from sys import argv
from gensim.models import Word2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree

from gensim import utils, matutils



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




EPOCHS = 51
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
	for line in test_file:
		try:
			line = line.strip().split("\t") 
			
			#slave = matutils.unitvec(array(infer_bug_vector(line[1].split(" ") + line[2].split(" "), int(dimensions)))).astype(REAL)
			#master = matutils.unitvec(array(infer_bug_vector(line[5].split(" ") + line[6].split(" "), int(dimensions)))).astype(REAL)
			slave_vec = array(infer_bug_vector(line[2].split(" ") +   line[3].split(" "), int(dimensions), model, algorithm))
			slave = matutils.unitvec(slave_vec).astype(REAL)
			master_vec = array(infer_bug_vector(line[6].split(" ") + line[7].split(" "), int(dimensions), model, algorithm))
			master = matutils.unitvec(master_vec).astype(REAL)
			actual_labels.append(int(float(line[4])))
			actual_labels.append(int(float(line[4])))
			slave_vectors.append(slave)
			total_vectors.append(slave)
			total_vectors.append(master)
		except IndexError:
			print line
	print "Vectors Formed for epoch", epoch
	slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)



	dists = dot(slave_vectors,total_vectors.T)
	#print "distances done for epoch", epoch
	div = [1,5,10,20,40,60,80,100,500,1000]
	accuracy = [0.0]*len(div)
	result_file = file("Office_Results_"+ algorithm +".txt","a+")

	for i in xrange(len(dists)):
		dist = dists[i]
		best = matutils.argsort(dist, topn=10001, reverse=True)

		best = best[1:]
		actual_label = actual_labels[i*2]
		pred_labels = map(lambda x: int(float(actual_labels[x])),best)

		for i in xrange(len(div)):
			if int(float(actual_label)) in pred_labels[:div[i]]:
				accuracy[i]+=1
	print accuracy
	accuracy = list(np.array(accuracy)/float(len(slave_vectors)))
	accuracy = map(lambda x: str(round(x, 3)), accuracy)
	result = algorithm + "\t" +str(dimensions) + "\t" + str(epoch)+ "\t" + " ".join(accuracy)+"\n"

	result_file.write(result)
	result_file.close()
	test_file.close()





def __main__():
	algorithm = argv[1]
	dimensions = argv[2]
	test_file = argv[3]
	for i in xrange(1, EPOCHS, 1):
		val(algorithm, dimensions, test_file, str(i))
__main__()	
