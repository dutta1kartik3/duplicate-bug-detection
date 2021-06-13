from sys import argv
from gensim.models import Doc2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc

dimensions = argv[1]
epoch = argv[2]
algo = int(argv[3])
test_file = file(argv[4])

algos = {0:"DBOW",1:"PVDM"}

model_filename = "/lustre/amar/Word2Vec_models/Doc2Vec_model_algo"+ str(algo) +"__dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".doc2vec"


#model_filename = "Doc2Vec.model.doc2vec"


model = Doc2Vec.load(model_filename)
print "Model Loaded"
slave_vectors = []
master_vectors = []
total_vectors = []
actual_labels = []

def infer_bug_vector(words):
	result = array([0.0]*int(dimensions))
	count = 1
	for word in words:
		if word in model:
			result+=model[word]
			count+=1
	return result/count


for line in test_file:
	try:
		line = line.strip().split("\t")
                slave = matutils.unitvec(array(model.infer_vector(line[1].split(" ") +   line[2].split(" ")))).astype(REAL)
                master = matutils.unitvec(array(model.infer_vector(line[5].split(" ") + line[6].split(" ")))).astype(REAL)
		actual_labels.append(set([ line[3], line[-1]]))
		actual_labels.append(set([ line[3], line[-1]]))	
		slave_vectors.append(slave)
		total_vectors.append(slave)
		total_vectors.append(master)
	except IndexError:
		print line
	#master_vectors.append(model.infer_vector(line[4]))

slave_vectors = np.nan_to_num(slave_vectors)
print "$$$$$$$$$$$$$$$$$$$$$$$$$"
print "Vectors Formed"
slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)

tree = KDTree(total_vectors, leaf_size=30) 

#dists = dot(slave_vectors,total_vectors.T)

dist, inds = tree.query(slave_vectors, k=100)
print "distances done"
prediction_file = open("/lustre/amar/predictions/prediction_doc2vec_dimensions_ " + str(dimensions) + "_algo_"+ algos[algo] + "_epoch_" + str(epoch)+"_euclidean.csv","w+")
div = [1,5,10,20,40,60,80,100]
accuracy = [0.0]*len(div)
result_file = file("Results.txt","a+")
for i in xrange(len(inds)):
	best = inds[i]

	best = best[1:]
	actual_label = actual_labels[i*2]
	pred_labels = map(lambda x: actual_labels[x],best)
	prediction_file.write(str(actual_label)+"\t"+str(pred_labels)+"\n")
	#div = [20,40,60,80,100]
	for i in xrange(len(div)):
                pred_label_set = set([item for sublist in pred_labels[:div[i]+1] for item in sublist])		
		if actual_label.intersection(pred_label_set)!=set([]):
			accuracy[i]+=1
			


result = "dimensions: " + str(dimensions) + " algo: "+ algos[algo] + " epoch: " + str(epoch)+ " Euclidean_" + str(accuracy)+"\n"

result_file.write(result)
result_file.close()
prediction_file.close()
