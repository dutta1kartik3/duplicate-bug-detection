from sys import argv
from gensim.models import Word2Vec, Doc2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc

dimensions = argv[1]
epoch = argv[2]
algo = argv[3]
test_file = file(argv[4])

algos = {0:"CBOW",1:"SG"}
#Word2Vec_model_algo1__dimensions_256_epoch_0.word2vec

model_filename = "/lustre/amar/Word2VecModels/"+algo+"_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec"
#else:
#	model_filename = "/lustre/amar/Word2Vec_models/Word2Vec_model_algo1__dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec"

#model_filename = "Doc2Vec.model.doc2vec"


model = Word2Vec.load(model_filename)
print "Model Loaded"
slave_vectors = []
master_vectors = []
total_vectors = []
actual_labels = []

def infer_bug_vector(words):
	result = array([0.0]*int(dimensions))
	count = 0
	for word in words:
		if word in model:
			result+= model[word]
			count+=1
	return result/count

for line in test_file:
	break

for line in test_file:
	try:
		line = line.strip().split("\t")
		slave = matutils.unitvec(array(infer_bug_vector(line[2].split(" ") +   line[3].split(" ")))).astype(REAL)
		master = matutils.unitvec(array(infer_bug_vector(line[6].split(" ") + line[7].split(" ")))).astype(REAL)
		actual_labels.append(line[5])
		actual_labels.append(line[5])	
		slave_vectors.append(slave)
		total_vectors.append(slave)
		total_vectors.append(master)
	except IndexError:
		print line
	#master_vectors.append(model.infer_vector(line[4]))


print "$$$$$$$$$$$$$$$$$$$$$$$$$"
print "Vectors Formed"
slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)



dists = dot(slave_vectors,total_vectors.T)
print "distances done"
#prediction_file = open("prediction_doc2vec_dimensions: " + str(dimensions) + " algo: "+ algos[algo] + " epoch: " + str(epoch)+".csv","w+")
div = range(1,1001,10)
accuracy = [0.0]*len(div)
result_file = file("Results.txt","a+")
for i in xrange(len(dists)):
	dist = dists[i]
	best = matutils.argsort(dist, topn=100, reverse=True)

	best = best[1:]
	actual_label = actual_labels[i*2]
	pred_labels = map(lambda x: int(actual_labels[x]),best)
	#prediction_file.write(str(actual_label)+"\t"+str(pred_labels)+"\n")
	#div = [20,40,60,80,100]
	for i in xrange(len(div)):
		if int(actual_label) in pred_labels[:div[i]]:
			accuracy[i]+=1
			


result = "\ndimensions: " + str(dimensions) + " algo: "+ algo + " epoch: " + str(epoch)+ " " + " " + str(accuracy)+ "\n"

result_file.write(result)
result_file.close()
#prediction_file.close()
