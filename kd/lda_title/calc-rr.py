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
from scipy import spatial

K = [1,5,10,20,40,60,80,100,500,1000] # Different values to be used in recall rate calculation
final_ans = [0 for x in range(0,len(K))] #Answer for different values of K, follows same index as above list
count_K = {} #Dictionary to map b/w K value(Used in RR@K) and which index it's located at
count = 0
for i in K:
        count_K[i] = count #count_K = {1:0,5:1 ...}
        count = count + 1

#Define some dictionaries relating pair id's and feature vector that we need to compute
data_pattern = [('panda_id',0,int), ('bug1_title',1,str),('bug1_desc',2,str),('bug1_priority',3,str), ('bug1_product',4,str),('bug1_component',5,str), ('bug1_severity',6,str), ('bug1_timestamp',7,str), ('bug2_title',8,str),('bug2_desc',9,str),('bug2_priority',10,str), ('bug2_product',11,str),('bug2_component',12,str), ('bug2_severity',13,str), ('bug2_timestamp',14,str),('label',15,int)] #pattern of a single entry

# bug_id_x        title_x description_x   dup_id_x        bug_id_y        title_y description_y   dup_id_y        bug_id_x        priority_x      product_x       component_x     sev_x   dup_id_x        creation_ts_x   bug_id_y        priority_y      product_y       component_y     sev_y   dup_id_y        creation_ts_y

rr_test_data_pattern = [('panda_id',0,int), ('bug_id',1,int), ('bug_title',2,str),('bug_desc',3,str),('master_id',4,int), ('master_id',5,int), ('master_title',6,str), ('master_desc',7,str),('garbage1',8,str), ('bug_id',9,int), ('bug_priority',10,str), ('bug_product',11,str), ('bug_component',12,str),('bug_severity',13,str),('master_id',14,float), ('bug_timestamp',15,str),('master_id',16,int), ('master_priority',17,str), ('master_product',18,str),('master_component',19,str),('master_severity',20,str),('garbage2',21,str),('master_timestamp',22,str)] #pattern of a single entry

lda_index1 = {} # Mapping b/w lda vectors and index
lda_index2 = {} # Mapping b/w lda vectors and index

def cosine_sim(p, q):
    p = [x[1] for x in p]
    q = [x[1] for x in q]
    return spatial.distance.cosine(p, q)

def readData(file_name):
        dups = open(file_name)
        dups_list = dups.readlines()
        dups_list.pop(0) #First element is the pattern header
        dups.close()
        return dups_list

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

def prepareDataRR(train_data,query_idx, lda_title_model,title_dict):
# Each feature vector is present in the order mentioned in Table I of the paper
    global rr_test_data_pattern
    features = [] # List of preprocessed documents that is returned
    gt = []
    portion_query = train_data[query_idx].split('\t')
    assert(len(portion_query)==len(rr_test_data_pattern)) #To ensure all entries match the data pattern
    for j in xrange(0,len(portion_query)):
          portion_query[j] = rr_test_data_pattern[j][2](portion_query[j]) #Convert each value to proper data type as mentioned
          if(j == len(portion_query) -1):
            portion_query[j] = portion_query[j].strip()
          if(rr_test_data_pattern[j][2] is float):
            portion_query[j] = int(portion_query[j])
    gt.append(1)
    gamma = []
    
    #Next 3 features are Hellinger distances of topics from title, desc and comb. LDA's
    q1 = lda_title_model[title_dict.doc2bow(preProcess(portion_query[2] + ' ' + portion_query[3]))]
    q2 = lda_title_model[title_dict.doc2bow(preProcess(portion_query[6]+' '+portion_query[7]))]
    
    gamma.append(cosine_sim(q1,q2))
    
    for report in xrange(0,len(train_data)):
          temp1 = [] #Feature vector to be appended in features list
          temp2 = []
          #print report
          portion = train_data[report].split('\t')
          if(report == query_idx):
                 continue
          if(report not in lda_index1 or report not in lda_index2):
              assert(len(portion)==len(rr_test_data_pattern)) #To ensure all entries match the data pattern
              for j in xrange(0,len(portion)):
                portion[j] = rr_test_data_pattern[j][2](portion[j]) #Convert each value to proper data type as mentioned
                if(j == len(portion) -1):
                  portion[j] = portion[j].strip()
                if(rr_test_data_pattern[j][2] is float):
                  portion[j] = int(portion[j])

              

            #Next 3 features are Hellinger distances of topics from title, desc and comb. LDA's
              
              lda_index1[report] = lda_title_model[title_dict.doc2bow(preProcess(portion[2]+' '+portion[3]))]
              lda_index2[report] = lda_title_model[title_dict.doc2bow(preProcess(portion[2]+' '+portion[3]))]
              
              temp1.append(cosine_sim(q1,lda_index1[report]))

              temp2.append(cosine_sim(q2,lda_index2[report]))
              
          else:
                temp1.append(cosine_sim(q1,lda_index1[report]))

                temp2.append(cosine_sim(q2,lda_index2[report]))

          
          features.append(temp1)
          features.append(temp2)
          gt.append(0)
          gt.append(0)

    return features,gt

# create English stop words list
stop_words = readData('stop-words.txt') #1 extra line at the top to take care of pop, list used by  Alipour et al.
for i in xrange(0,len(stop_words)):
        stop_words[i] = stop_words[i].strip() #Remove whitespaces here
en_stop = frozenset(stop_words)
    
lda_title_model = gensim.models.ldamulticore.LdaMulticore.load('LDA_Models/mozilla-lda-title')

with open('Data/MozillaLDATitleDict.data', 'r') as f:
        title_dict = pickle.load(f)

rr_test_data = readData('DupsTestWithMetaData5000.csv')
rr_dtype = [('not-duplicate',int ), ('duplicate', int)]

start = [0,500,1000,1500,2000,2500,3000,3500,4000,4500]
finish = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000]
assert(len(start)==len(finish))
for i in xrange(0,len(rr_test_data)):
    print i
    scores, ground_truth = prepareDataRR(rr_test_data,i, lda_title_model, title_dict)
    
    #eta = joblib.load('LR_Models/mozilla_LR.pkl') 
    #scores = eta.predict_proba(np.matrix(rr_proc_test_data))
    #assert(ground_truth[0]==1)
    dump = []
    for k in xrange(0,len(scores)):
       dump.append(scores[k][0])
    junk = np.array(dump)
    junk2 = np.argsort(junk) # 0th idx is the actual value we want, checkout where it lands after sorting based on prob of not being duplicate
    counter = 0
    ticker = 0
    #print junk2
    for j in xrange(0,len(junk2)):
      counter = counter + 1
      if(junk2[j]==0):
        ticker = 1
        #print counter

      if(counter > max(K)): #We only need to scan top 1000 entries in case
        break

      if(counter in count_K):
        final_ans[count_K[counter]] = final_ans[count_K[counter]] + ticker
    
    #use scores from idx 0 here
#print final_ans

print '\n'
print len(rr_test_data)
print K
print final_ans #Without Dividing by count_relevant_queries
for i in xrange(0,len(final_ans)):
        final_ans[i] = final_ans[i] + 0.0 #Conver to float
        final_ans[i] = final_ans[i] / len(rr_test_data)

print final_ans # The actual RR .... 
'''
for i in xrange(0,len(final_ans)):
  f.write(str(K[i]))
  f.write(' ')
f.write('\n')
for i in xrange(0,len(final_ans)):
  f.write(str(final_ans[i]))  # python will convert \n to os.linesep
  f.write(' ')
f.close() '''
