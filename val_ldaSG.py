from gensim.models.ldamodel import LdaModel
from gensim import corpora, models, similarities
from sys import argv
from numpy import  array, float32 as REAL,dot
import numpy as np
from gensim import utils, matutils
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.porter import PorterStemmer
import logging
import itertools
import re
import gensim
from sys import argv
from gensim.models import Word2Vec
from numpy import  array, float32 as REAL,dot
import numpy as np
from sklearn.neighbors import KDTree

from gensim import utils, matutils



logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO

stemmer = PorterStemmer()

num_topics = int(argv[1])
iters = int(argv[2])
test_file = file(argv[3])
dimensions = argv[4]
epoch = argv[5]
algo = int(argv[6])
p = int(argv[7])
type = argv[8]


model_filename = "/lustre/amar/lda/Dup_LdaModel_topics_" + str(num_topics) + "_passes_"+str(iters) + ".model"


def tokenize(text):
	return [token for token in text.split(" ") if token not in STOPWORDS]


#150k_LdaModel_topics_200_passes_100.model

model = LdaModel.load(model_filename)  



def infer_bug_vector_lda(s):
	s = tokenize(s)
	s = map(stemmer.stem, s)
	try:
		bow_vector = dictionary.doc2bow(s)
		lda_vector = model[bow_vector]
		vector = [0.0]*num_topics
		for i in lda_vector:
			vector[i[0]] = i[1]
		#print vector
		return vector
	except IndexError:
		print "In Infer"
		return " "



print "Model Loaded"
slave_vectors = []
master_vectors = []
total_vectors = []
actual_labels = []


dictionary = corpora.Dictionary.load("/lustre/amar/lda/LDADictOnlyDups_200000.dict")
count = 0

#['756', 'abbr not handled correctly', 'not sure if this is a parser or a content model problem  but abbr seems to hose things  nglayout inserts a newline before any abbr content  and may hose nearby inline markup ', '1358', '1358', ' 4xp  tooltips do not work', 'tooltips should be supported on any element that has a title attribute  especially a s and img s  in the case of an img with an alt attribute and a title attribute  the title should be used for the tooptip ', '1995']


for line in test_file:
	count+=1
	if count%500 == 0:
		print count, "vectors formed"
	try:
		line = line.strip().split("\t")
                actual_labels.append(set([line[3], line[-1]]))
                actual_labels.append(set([line[3], line[-1]]))
	        
		line = map(lambda x : re.sub('[+<>{}\[\](),&^%$#@!-_*.]', ' ', x), line )
		line = map(lambda x : re.sub(r'\b[0-9]+\b\s*', '', x), line)
		line = map(lambda x: " ".join(x.split()), line)

	        #line = re.sub('[+<>{}\[\](),&^%$#@!-_*.]', ' ', line)
	        #line = re.sub(r'\b[0-9]+\b\s*', '', line)
	        
		#line = line.strip().split("\t")
		slave = array(infer_bug_vector_lda(line[1] + " " + line[2]))

		
		master = array(infer_bug_vector_lda(line[5] + " " + line[6]))
		#actual_labels.append(set([line[3], line[-1]]))
		#actual_labels.append(set([line[3], line[-1]]))	
		slave_vectors.append(slave)
		total_vectors.append(slave)
		total_vectors.append(master)
	except IndexError:
		print line


print "$$$$$$$$$$$$$$$$$$$$$$$$$"
print "Vectors Formed"
slave_vectors,total_vectors = array(slave_vectors), array(total_vectors)



dists = dot(slave_vectors,total_vectors.T)
print "distances done"
prediction_file = open("DupLdaprediction_topics: " + str(num_topics) + "_passes_"+ str(iters)  + ".csv","w+")

div = [1,5,10,20,40,60,80,100,200,300,400,500,600,700,800,900,1000]
accuracy = [0.0]*len(div)
result_file = file("Results.txt","a+")
count = 0
lda_predictions = []
for i in xrange(len(dists)):
	count+=1
	if count%500==0:
		print accuracy
		print count, "done"

	dist = dists[i]
	best = matutils.argsort(dist, topn=1000, reverse=True)

	best = best[1:]
	lda_predictions.append(best)


print "LDA DONE"

algos = {0:"CBOW",1:"SG"}

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

dist, inds = tree.query(slave_vectors, k=15000)
print "distances done"
prediction_file = open("/lustre/amar/predictions/prediction_doc2vec_dimensions_ " + str(dimensions) + "_algo_"+ algos[algo] + "_epoch_" + str(epoch)+"_euclidean.csv","w+")
div = [1,5,10,20,40,60,80,100,200,300,400,500,600,700,800,900,1000]
accuracy = [0.0]*len(div)
result_file = file("Results.txt","a+")
for i in xrange(len(inds)):
	best = inds[i]
	best = filter(lambda x:x in pred_label_set[i], best)

	actual_label = actual_labels[i*2]
	pred_labels = map(lambda x: actual_labels[x],best)
	prediction_file.write(str(actual_label)+"\t"+str(pred_labels)+"\n")
	#div = [20,40,60,80,100]
	for i in xrange(len(div)):
                pred_label_set = set([item for sublist in pred_labels[:div[i]+1] for item in sublist])		
		if actual_label.intersection(pred_label_set)!=set([]):
			accuracy[i]+=1
			
accuracy = array(accuracy)/15000.0
print accuracy
accuracy = np.around(accuracy, decimals = 3)

result = "dimensions: " + str(dimensions) + " algo: "+ algos[algo] + " epoch: " + str(epoch)+ " Euclidean_"+ str(accuracy) +"\n"

result_file.write(result)
result_file.close()
prediction_file.close()
