#Code to implement Klien's et al's Paper
'''Conventions used in this code and data
Label 0 means these are not duplicates
Label 1 means these are duplicates
Product 0 means they two samples don't belong to same product
Product 1 means two samples belong to same product
Priority 0 means the 2 sample have diff. priority
Priority 1 means the 2 samples have same priority
Component 0 means the 2 samples involve diff. components 
Component 1 means the 2 samples involve same components 
'''

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from sklearn.linear_model import LogisticRegressionCV
from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import gensim
from datetime import datetime
from sklearn.externals import joblib
import time
import csv
import pickle

K = [1,5,10,20,40,60,80,100,500,1000] # Different values to be used in recall rate calculation
final_ans = [0 for x in range(0,len(K))] #Answer for different values of K, follows same index as above list
count_K = {} #Dictionary to map b/w K value(Used in RR@K) and which index it's located at
count = 0
for i in K:
        count_K[i] = count #count_K = {1:0,5:1 ...}
        count = count + 1

#Define some dictionaries relating pair id's and feature vector that we need to compute
data_pattern = [('panda_id',0,int), ('bug1_title',1,str),('bug1_desc',2,str),('bug1_priority',3,str), ('bug1_product',4,str),('bug1_component',5,str), ('bug1_severity',6,str), ('bug1_timestamp',7,str), ('bug2_title',8,str),('bug2_desc',9,str),('bug2_priority',10,str), ('bug2_product',11,str),('bug2_component',12,str), ('bug2_severity',13,str), ('bug2_timestamp',14,str),('label',15,int)] #pattern of a single entry

def hellinger(p, q): #Taken from https://gist.github.com/larsmans/3116927
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) /  np.sqrt(2)

def readData(file_name):
        dups = open(file_name)
        dups_list = dups.readlines()
        dups_list.pop(0) #First element is the pattern header
        dups.close()
        return dups_list

def findMinsDiff(tstamp1,tstamp2): #Return time diff in minutes from timestamps
	temp = tstamp1.split(' ')
	tstamp1 = temp[0] + ' ' + temp[1]

	temp = tstamp2.split(' ')
        tstamp2 = temp[0] + ' ' + temp[1]

	fmt = '%Y-%m-%d %H:%M:%S'
	d1 = datetime.strptime(tstamp1, fmt)
	d2 = datetime.strptime(tstamp2, fmt)

# Convert to Unix timestamp
	d1_ts = time.mktime(d1.timetuple())
	d2_ts = time.mktime(d2.timetuple())

# They are now in seconds, subtract and then divide by 60 to get minutes.
	return abs(int(d2_ts-d1_ts) / 60)

def prepareDataLDA(raw_data): # Prepare and tokenize ... the data
	global data_pattern
	texts_title = [] # List of preprocessed documents that is returned
	texts_desc = []
	texts_comb = []
	for report in xrange(0,len(raw_data)): 
		print report
		portion = raw_data[report].split('\t')
        	assert(len(portion)==len(data_pattern)) #To ensure all entries match the data pattern
		for j in xrange(0,len(portion)):
	                portion[j] = data_pattern[j][2](portion[j]) #Convert each value to proper data type as mentioned
		texts_title.append(preProcess(portion[1] + ' ' + portion[8]))
		texts_desc.append(preProcess((portion[2] + ' ' + portion[9])))
		texts_comb.append(preProcess(portion[1] + ' ' + portion[8] + ' ' + portion[2] + ' ' + portion[9]))
			
	return texts_title, texts_desc, texts_comb

def preProcess(data_string):
	global en_stop
	tokenizer = RegexpTokenizer(r'\w+') #Create appropriate tokenizer
        p_stemmer = PorterStemmer() #Create object from Porter Stemmer
	#clean and tokenize document string
	raw = data_string.lower()
	tokens = tokenizer.tokenize(raw)
	# remove stop words from tokens
	stopped_tokens = [i for i in tokens if not i in en_stop]
	# stem tokens
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	return stemmed_tokens

def sortLDADist(vector1, vector2): # Return the first common topic between two topic vectors returned by lda
	topic_id = None
	dtype = [('topic_id', int),('score' ,int)] #used for sorting the array
	sorted_v1 = np.array(vector1, dtype=dtype)
        sorted_v1 = np.sort(sorted_v1,order='score')
        sorted_v1 = sorted_v1[::-1] #reverse  to make it sorted in descending order
	sorted_v2 = np.array(vector2, dtype=dtype)
        sorted_v2 = np.sort(sorted_v2,order='score')
        sorted_v2 = sorted_v2[::-1] #reverse  to make it sorted in descending order
	assert(len(sorted_v1) == len(sorted_v2))

	for i in xrange(0,len(sorted_v1)):
		if(sorted_v1[i][0]==sorted_v2[i][0]):
			topic_id = sorted_v1[i][0]
			break

	if(topic_id is None):
		return -1
	else:
		return topic_id

def prepareDataLReg(train_data, lda_title_model, lda_desc_model, lda_comb_model,title_dict, desc_dict, comb_dict):
# Each feature vector is present in the order mentioned in Table I of the paper
	global data_pattern
        features = [] # List of preprocessed documents that is returned
	gtruth = [] #Ground truth for the features
        for report in xrange(0,len(train_data)):
		temp = [] #Feature vector to be appended in features list
                print report
                portion = train_data[report].split('\t')
                assert(len(portion)==len(data_pattern)) #To ensure all entries match the data pattern
                for j in xrange(0,len(portion)):
                        portion[j] = data_pattern[j][2](portion[j]) #Convert each value to proper data type as mentioned

		#First 2 features are diff. in number of words in un-processed title and desc resp.
		temp.append(abs( len(portion[1].split(' ')) -len(portion[8].split(' '))) )
		temp.append(abs( len(portion[2].split(' ')) -len(portion[9].split(' '))) )

		#Next 2 features are number of shared words in processed title and desc resp.
		temp.append(len(frozenset(preProcess(portion[1])).intersection(frozenset(preProcess(portion[8])))))
		temp.append(len(frozenset(preProcess(portion[2])).intersection(frozenset(preProcess(portion[9])))))
		
		#Next 3 features are the first shared identical topic b/w title, desc and combined case
		temp.append(sortLDADist(lda_title_model[title_dict.doc2bow(preProcess(portion[1]))],lda_title_model[title_dict.doc2bow(preProcess(portion[8]))]))
		temp.append(sortLDADist(lda_desc_model[desc_dict.doc2bow(preProcess(portion[2]))],lda_desc_model[desc_dict.doc2bow(preProcess(portion[9]))]))
		temp.append(sortLDADist(lda_comb_model[comb_dict.doc2bow(preProcess(portion[1] + ' ' + portion[2]))],lda_comb_model[comb_dict.doc2bow(preProcess(portion[8]+' '+portion[9]))]))
		
		#Next 3 features are Hellinger distances of topics from title, desc and comb. LDA's
		temp.append(hellinger(lda_title_model[title_dict.doc2bow(preProcess(portion[1]))],lda_title_model[title_dict.doc2bow(preProcess(portion[8]))]))
		temp.append(hellinger(lda_desc_model[desc_dict.doc2bow(preProcess(portion[2]))],lda_desc_model[desc_dict.doc2bow(preProcess(portion[9]))]))
		temp.append(hellinger(lda_comb_model[comb_dict.doc2bow(preProcess(portion[1] + ' ' + portion[2]))],lda_comb_model[comb_dict.doc2bow(preProcess(portion[8]+' '+portion[9]))]))
		#Next feature is whether priority is same or not
                temp.append(1 if portion[3] == portion[10] else 0)
		#Next feature represents difference in minutes between 2 reports
                temp.append(findMinsDiff(portion[7],portion[14]))
		#Next feature is whether component is same or not
                temp.append(1 if portion[5] == portion[12] else 0)
		#Next feature is whether prodcut is same or not 
		temp.append(1 if portion[4] == portion[11] else 0)
		#Put ground truth in the class label
		gtruth.append(portion[15])
		features.append(temp)

	return features,gtruth

def trainLDA(texts): # Used for training LDA from pre-processed documents

	# turn our tokenized documents into a id <-> term dictionary
	dictionary = corpora.Dictionary(texts)
	# convert tokenized documents into a document-term matrix
	corpus = [dictionary.doc2bow(text) for text in texts]
	# generate LDA model
	ldamodel = gensim.models.ldamulticore.LdaMulticore(corpus, minimum_probability=None, num_topics=100, id2word = dictionary, eta=0.01,workers=1, iterations=10000,passes=5) 
	return ldamodel,dictionary

# create English stop words list
stop_words = readData('stop-words.txt') #1 extra line at the top to take care of pop, same stop word list used by  Alipour et al.
for i in xrange(0,len(stop_words)):
        stop_words[i] = stop_words[i].strip() #Remove whitespaces here
en_stop = frozenset(stop_words)
    
#Read Train Data
############################################################################
#train_data = readData('Datasets/ClassificationDupsTrainWithMetaData.csv')
#with open('Data/OfficePairTrain.data', 'w') as f:
#	pickle.dump(train_data, f)
############################################################################

with open('Data/OfficePairTrain.data', 'r') as f:
       train_data = pickle.load(f)

#To Prepare LDA Training Data
############################################################################
#train_lda = train_data[:] # Only keep titles from train data for title lda mode
#train_lda_title, train_lda_desc , train_lda_comb  = prepareDataLDA(train_lda)
#############################################################################

with open('Data/OfficeLDATitle.data', 'r') as f:
        train_lda_title = pickle.load(f)
	#pickle.dump(train_lda_title, f)
with open('Data/OfficeLDADesc.data', 'r') as f:
       	train_lda_desc = pickle.load(f)
	#pickle.dump(train_lda_desc,f)
with open('Data/OfficeLDAComb.data', 'r') as f:
	train_lda_comb = pickle.load(f)
	#pickle.dump(train_lda_comb,f)

#Train and Save Title LDA Model and corresponding dictionary
#######################################################
#lda_title_model,title_dict = trainLDA(train_lda_title)
#lda_title_model.save('LDA_Models/mozilla-lda-title')
#with open('Data/MozillaLDATitleDict.data', 'w') as f:
#        title_dict = pickle.load(f)
#	pickle.dump(title_dict,f)
#######################################################
lda_title_model = gensim.models.ldamulticore.LdaMulticore.load('LDA_Models/office-lda-title')

#Train and Save Desc LDA model and corresponding dict.
######################################################
#lda_desc_model,desc_dict = trainLDA(train_lda_desc)
#lda_desc_model.save('LDA_Models/mozilla-lda-desc')
#with open('Data/MozillaLDADescDict.data', 'w') as f:
#	pickle.dump(desc_dict,f)
######################################################
lda_desc_model = gensim.models.ldamulticore.LdaMulticore.load('LDA_Models/office-lda-desc')

#Train and Save Title + Desc LDA model and corresponding dict.
######################################################
#lda_comb_model,comb_dict = trainLDA(train_lda_comb)
#lda_comb_model.save('LDA_Models/mozilla-lda-comb')
#with open('Data/MozillaLDACombDict.data', 'w') as f:
        #pickle.dump(comb_dict,f)
######################################################
lda_comb_model =  gensim.models.ldamulticore.LdaMulticore.load('LDA_Models/office-lda-comb')

#Load the 3 Codebooks used for LDA
with open('Data/OfficeLDATitleDict.data', 'r') as f:
        title_dict = pickle.load(f)
with open('Data/OfficeLDADescDict.data', 'r') as f:
        desc_dict = pickle.load(f)
with open('Data/OfficeLDACombDict.data', 'r') as f:
        comb_dict = pickle.load(f)

#To Calculate the train features
#train_data, train_data_labels = prepareDataLReg(train_data, lda_title_model, lda_desc_model, lda_comb_model,title_dict,desc_dict,comb_dict)

with open('Data/OfficePairTrainFeatures.data', 'r') as f:
        #pickle.dump(train_data, f)
        train_data = pickle.load(f)

with open('Data/OfficePairTrainLabels.data', 'r') as f:
       #pickle.dump(train_data_labels, f)
        train_data_labels = pickle.load(f)
	
#Read Test Data
#########################################################################
#test_data = readData('Datasets/ClassificationOfficeDupsTestWithMetaData.csv')
#test_data_labels = None
#########################################################################
with open('Data/OfficePairTest.data', 'r') as f:
       test_data = pickle.load(f)

#To calculate the test features
#test_data, test_data_labels = prepareDataLReg(test_data, lda_title_model, lda_desc_model, lda_comb_model,title_dict,desc_dict,comb_dict)

with open('Data/OfficePairTestFeatures.data', 'r') as f:
	test_data = pickle.load(f)

with open('Data/OfficePairTestLabels.data', 'r') as f:
	test_data_labels = pickle.load(f)

#Now Train on LReg. Classifier

#Code to conduct the actual training
######################################

#train_data = np.matrix(train_data)
#eta = LogisticRegressionCV(n_jobs=4, verbose=1,max_iter=100000,solver='newton-cg',Cs=1000)
#eta.fit(train_data,train_data_labels)
########################################
eta = joblib.load('LR_Models/office_LR.pkl')
scores = eta.predict_proba(np.matrix(test_data))

write = []
write.append(['y_prob_dupl','y_gt'])
for i in xrange(0,len(scores)):
	temp = []
	temp.append(scores[i][1])
	temp.append(test_data_labels[i])
	write.append(temp)

with open("roc-office.csv", "w") as f:
        writer = csv.writer(f,delimiter='\t')
        writer.writerows(write)

