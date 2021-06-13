#Implementation of Yang et al.'s work
import numpy as np
import snowballstemmer as sst
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import gensim
from sklearn.metrics.pairwise import cosine_similarity


K = [1,5,10,20,40,60,80,100,500,1000] # Different values to be used in recall rate calculation
final_ans = [0 for x in range(0,len(K))] #Answer for different values of K, follows same index as above list
count_K = {} #Dictionary to map b/w K value(Used in RR@K) and which index it's located at
count = 0
for i in K:
        count_K[i] = count #count_K = {1:0,5:1 ...}
        count = count + 1

bid_tfidf = {} #Dictionary between a bug's bug_id and its title + desc tf-idf vector
bid_vec = {} # Same as above, only for storing doc_vec of title+desc
bid_pdcp = {} # Dictionary between bug_id and frozenset containing it's product and component info
bid_text_words = {} # Dict. b/w bug_id and the list of pre-processed words in that bug. For calculating idf
bid_mid = {} #Mapping b/w a bug's id and it's master id
vocab = frozenset(['']) # List of all possible words

def readData(file_name):
	dups = open(file_name)
	dups_list = dups.readlines()
	dups_list.pop(0) #First element is the pattern header
	dups.close()
	return dups_list

def keep_alpha_only(word_list):
	for i in xrange(0,len(word_list)):
        	word_list[i] = ''.join(char for char in word_list[i] if char.isalpha()) #Keep only alphabetic characters

def get_docvec(model,word_list): #Output Doc-Vector by avg. word2vec
	avg_vec = np.zeros(100)

	for word in word_list:
		if word not in model: #Word2vec only works for seen words
			continue
                if avg_vec is None:
                        avg_vec = model.wv[word]
                else:
                        avg_vec = avg_vec +  model.wv[word]

	avg_vec = avg_vec / len(word_list)
	return avg_vec
	

stop_words = readData('stop-words.txt') #1 extra line at the top to take care of pop, list used by  Alipour et al.
for i in xrange(0,len(stop_words)):
	stop_words[i] = stop_words[i].strip() #Remove whitespaces here
stop_words = frozenset(stop_words)

dups_list = readData('Datasets/DupsTestWithMetaData5000.csv') #DupsTestWithMetaData5000.csv, OfficeDupsTestWithMetaData.csv
word2vec_model = gensim.models.Word2Vec.load('word2vec/skipgram_model_dimensions_100_epoch_5.word2vec') #openoffice_skipgram_model_dimensions_100_epoch_5.word2vec 
count_relevant_queries = 0 #Count of number of relevant bug queries

data_pattern = [('panda_id',0,int), ('bug_id1' ,1,int),('bug_title',2,str),('bug_desc',3,str),('master_id1',4,int),('master_id2',5,int),('master_title',6,str),('master_desc',7,str), ('Empty String',8,str),('bug_id2',9,int), ('bug_priority',10,str), ('bug_product',11,str),('bug_component',12,str), ('bug_severity',13,str), ('master_id3',14,float),('master_id4',15,int),('master_priority',16,str), ('master_product',17,str),('master_component',18,str), ('master_severity',19,str), ('eof',20,str)] #pattern of a single entry

for report in xrange(0,len(dups_list)):
	print report
	portion = dups_list[report].split('\t')
	assert(len(portion)==len(data_pattern)) #To ensure all entries match the data pattern
	for j in xrange(0,len(portion)):
		portion[j] = data_pattern[j][2](portion[j]) #Convert each value to proper data type as mentioned
		if(type(portion[j]) is float):
			portion[j] = int(portion[j]) # For some reason master_id3 is float, should be int in csv file

	bid_mid[portion[1]] = portion[4] #Put Master bug for current bug
	count_relevant_queries = count_relevant_queries + 1
	bid_mid[portion[4]] = portion[4] #Master bug is master of itself, ignored while querying

	bid_pdcp[portion[1]] = frozenset([portion[11],portion[12]]) #Put product and component info into frozenset for cmp.
	bid_pdcp[portion[4]] = frozenset([portion[17],portion[18]])

	bug_text = portion[2] + ' ' + portion[3] #Combine Desc and title into one document (text) field as per paper
	bug_text = bug_text.strip()
	master_text = portion[6] + ' ' + portion[7]
	master_text = master_text.strip()

	bug_text_tokens = word_tokenize(bug_text) #Tokenize, remove stop words
	master_text_tokens = word_tokenize(master_text)
	bug_text_filtered = [w for w in bug_text_tokens if not w in stop_words]
	master_text_filtered = [w for w in master_text_tokens if not w in stop_words]
	keep_alpha_only(bug_text_filtered) #Keep only alphabetic characters
	keep_alpha_only(master_text_filtered)
	stemmer = sst.stemmer('english') #Do Snowball stemming
	bug_text_stemmed = stemmer.stemWords(bug_text_filtered)
	master_text_stemmed = stemmer.stemWords(master_text_filtered)
	vocab =	vocab.union(frozenset(bug_text_stemmed)) # Add newer words to vocabulary
	vocab =	vocab.union(frozenset(master_text_stemmed))

	bid_text_words[portion[1]] = bug_text_stemmed # Used for calculating tf-idf
	bid_text_words[portion[4]] = master_text_stemmed
	bid_vec[portion[1]] = get_docvec(word2vec_model,bug_text_stemmed) #Store bug_id with its textual docvec
	bid_vec[portion[4]] = get_docvec(word2vec_model,master_text_stemmed) #Store master with its textual docvec


#Need to split bid_text_words to 2 iterable lists
bug_id = bid_text_words.keys()
documents = bid_text_words.values() # Need to convert this into a list of strings from list of lists
for i in xrange(0,len(documents)):
	documents[i] = ' '.join(documents[i])
	#documents.pop(0)

sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=True,  analyzer='word')
#Here sklearn learns vocab itself and excludes words with fq. = 0, analyzes at word level and only considers alphabetic words as tokens
#documents = np.array(documents).reshape((len(documents), 1))# Using python lists causes errors
sklearn_repr = sklearn_tfidf.fit_transform(documents) #Creates tf-idf vector for all documents in same order as the list inputs the documents, stored at row i for document at idx i

for i in xrange(0,len(bug_id)): #Put tf-idf vectors in the appropriate dictionary
	print i
	bid_tfidf[bug_id[i]] = sklearn_repr[i]

#Now to calculate RR@K
for bID in bid_mid:
	print bID
	if(bid_mid[bID]==bID): #Don't use master_bug's in the database to calculate recall rate
		continue
	score = [] #Store scores for a query, for all the logs in db
	for bug in bid_mid:
		if(bug == bID): # Queries score with itself is always highest, ignore
			continue

		score1 = cosine_similarity(bid_tfidf[bID], bid_tfidf[bug]) # Compute cosine similarity b/w doc tf-idf vectors
		score2 = cosine_similarity(bid_vec[bID], bid_vec[bug]) #Compute cosine similarity b/w document vectors
		score3 = len(bid_pdcp[bID].intersection(bid_pdcp[bug])) / len(bid_pdcp[bID].union(bid_pdcp[bug])) # Compute Jaccard similarity using the product and component info

		total = (score1 + score2)*score3
		score.append((bug,total))

	dtype = [('bug_id', int),('value' ,int)] #used for sorting the array
	sorted_score = np.array(score, dtype=dtype)
	sorted_score = np.sort(sorted_score,order='value')
	sorted_score = sorted_score[::-1] #reverse  to make it sorted in descending order
	count  = 0
	ticker = 0
	for result in sorted_score:
		if(result[0] == bid_mid[bID]): # Results bug_id (0th idx), matches its corresponding master id
			ticker = 1
		
		count = count + 1

		if(count > max(K)): #We only need to scan top 1000 entries in case
			break

		if(count in count_K):
			final_ans[count_K[count]] = final_ans[count_K[count]] + ticker #say count is 100, then count_K[100] = 7,idx of 100 in list K, hence update final_ans[7]

print '\n'
print count_relevant_queries
print K
print final_ans #Without Dividing by count_relevant_queries
for i in xrange(0,len(final_ans)):
	final_ans[i] = final_ans[i] + 0.0 #Conver to float
	final_ans[i] = final_ans[i]/ count_relevant_queries

print final_ans # The actual RR ....
