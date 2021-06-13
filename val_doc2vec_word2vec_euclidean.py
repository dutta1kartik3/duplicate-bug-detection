from sys import argv
from gensim.models import Word2Vec, Doc2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree


from gensim import utils, matutils  # utility fnc for pickling, common scipy operations etc
model_type = argv[1]
dimensions = argv[2]
source = int(argv[3])
algo = int(argv[4])
#test_file = file(argv[5])





def infer_bug_vector(words):
	#print words
	if model_type == "Doc2Vec":
		print "In Doc2vec"
		return model.infer_vector(words)
	
	result = array([0.0]*int(dimensions))
	count = 0
	for word in words:
		if word in model:
			result+= model[word]
		        count+=1
	return result/count



def validate(model_filename, result_file, epoch):
	test_file = file(argv[5])
	count = 0
	print "Model Loaded"
	slave_vectors = []
	master_vectors = []
	total_vectors = []
	actual_labels = []
	for line in test_file:
		try:

			line = line.strip().split("\t")
		        slave_text = ""
			master_text = ""
			if source == 0: # Both Titles and Desc
				slave_text = line[1].split(" ") +   line[2].split(" ")
				master_text = line[5].split(" ") + line[6].split(" ")
			elif source == 1:
				slave_text = line[1].split(" ")
				master_test = line[5].split(" ")
			elif source == 2:
				slave_text = line[2].split(" ")
				master_text = line[6].split(" ")			
			
			slave = matutils.unitvec(array(infer_bug_vector(slave_text))).astype(REAL)
			master = matutils.unitvec(array(infer_bug_vector(master_text))).astype(REAL)
			actual_labels.append(line[3])
			actual_labels.append(line[3])	
			slave_vectors.append(slave)
			total_vectors.append(slave)
			total_vectors.append(master)
			count+=1

		except IndexError:
			print line
	

	#print "$$$$$$$$$$$$$$$$$$$$$$$$$"
	#print "Vectors Formed"
	#slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)
	#dists = dot(slave_vectors,total_vectors.T)
	slave_vectors = np.nan_to_num(slave_vectors)
	print "$$$$$$$$$$$$$$$$$$$$$$$$$"
	print "Vectors Formed"
	slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)

	tree = KDTree(total_vectors, leaf_size=30)

	#dists = dot(slave_vectors,total_vectors.T)

	dists, inds = tree.query(slave_vectors, k=1001)




	print dists.shape
	print "distances done"	
	div = range(0,1001,10)
	div[0]  = 1
	accuracy = [0.0]*len(div)
	
	#result_fiile = file("Results.txt","a+")
	for i in xrange(len(dists)):
		#dist = dists[i]
		#best = matutils.argsort(dist, topn=10002, reverse=True)
		best = inds[i]
		#print i, best[:10]
		if 2*i in best:
			best = list(best)
			best.remove(2*i)
			best = np.array(best)
		#print list(best)[:10]
		#r = raw_input()
		actual_label = actual_labels[i*2]
		pred_labels = map(lambda x: int(actual_labels[x]),best)
		#r = raw_input()
	
		for i in xrange(len(div)):
			if int(actual_label) in pred_labels[:div[i]]:
				accuracy[i]+=1
			

	#print accuracy
	#print count
	accuracy = np.array(accuracy)/len(slave_vectors)
	r = str(epoch) + "\t" + "\t".join(map(str,list(accuracy))) + "\n"
	result_file.write(r)



algos = {0:"CBOW",1:"SG"}
#Titles_Word2Vec_model_algo1__dimensions_64_epoch_8_ng_5.word2vec
data = {0:"Both",1:"Titles", 2:"Desc"}

result_file = file("/lustre/amar/"+ model_type + "_results/Euclidean_Results_" + data[source] + "_" + model_type+ "_model_algo" + str(algo)+ "__dimensions_" + str(dimensions) + "_ng_5.tsv", "w+")

r = "0\t1\t" + "\t".join(map(str, range(10,1001,10))) + "\n"
result_file.write(r)


for epoch in range(0, 50):
#for epoch in [25]:	
	
	model_filename =  "/lustre/amar/"+ model_type + "_models/" + data[source] + "_" + model_type+ "_model_algo" + str(algo)+ "__dimensions_" + str(dimensions) + "_epoch_" + str(epoch) + "_ng_5." + model_type.lower()
	print model_filename	
	if model_type == "Word2Vec":	
		model = Word2Vec.load(model_filename)
	elif model_type == "Doc2Vec":
		model = Doc2Vec.load(model_filename)
	validate(model_filename, result_file, epoch)


result_file.close()
