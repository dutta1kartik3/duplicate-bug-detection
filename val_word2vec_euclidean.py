from sys import argv
from gensim.models import Word2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree

from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc

dimensions = argv[1]
epoch = argv[2]
algo = int(argv[3])
test_file = file(argv[4])
p = int(argv[5])
type = argv[6] # 0 for nothing and 1 for titles
val_test = argv[7]
#ng = argv[7]

algos = {0:"CBOW",1:"SG"}
data = {0: "All", 1:"Title"}
if type == '1':
	model_filename = "/lustre/amar/Word2Vec_models/Titles_"
else:
	model_filename = "/lustre/amar/Word2Vec_models/"

if algo == 1:
	model_filename += "Word2Vec_model_algo1__dimensions_"+str(dimensions)+"_epoch_"+str(epoch) + ".word2vec"
else:
	model_filename += "Word2Vec_model_algo0__dimensions_"+str(dimensions)+"_epoch_"+str(epoch) +".word2vec"

#model_filename = "Doc2Vec.model.doc2vec"
print model_filename

model = Word2Vec.load(model_filename)
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
		slave = array(infer_bug_vector(line[1].split(" ") +   line[2].split(" ")))
		master = array(infer_bug_vector(line[5].split(" ") + line[6].split(" ")))
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

dist, inds = tree.query(slave_vectors, k=7000)
print "distances done"
prediction_file = open("/lustre/amar/predictions/prediction_doc2vec_dimensions_ " + str(dimensions) + "_algo_"+ algos[algo] + "_epoch_" + str(epoch)+"_euclidean.csv","w+")
div = [1,5,10,20,40,60,80,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000]
accuracy = [0.0]*len(div)
result_file = file("Results.txt","a+")
for i in xrange(len(inds)):
	best = list(inds[i])
	if i*2 in best:
		best.remove(i*2)
	actual_label = actual_labels[i*2]
	pred_labels = map(lambda x: actual_labels[x],best)
	prediction_file.write(str(actual_label)+"\t"+str(pred_labels)+"\n")
	#div = [20,40,60,80,100]
	for i in xrange(len(div)):
		pred_label_set = []
		for k in pred_labels[:div[i]]:
			for j in list(k):
				pred_label_set.append(j)
                #pred_label_set = set([item for sublist in pred_labels[:div[i]] for item in sublist])		
		#print set(actual_label) , pred_label_set
		if set(actual_label).intersection(pred_label_set)!=set([]):
			accuracy[i]+=1
			
accuracy = array(accuracy)/float(15000.0)
print accuracy
accuracy = np.around(accuracy, decimals = 3)

result = " dimensions:" + str(dimensions) + " algo: "+ algos[algo] + " epoch: " + str(epoch)+ "Euclidean_"+ "_data_"+ data[int(type)]+"_" + str(accuracy) +"\n"

result_file.write(result)
result_file.close()
prediction_file.close()
