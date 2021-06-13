#Code to implement Sureka's et al work
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#(http://scikit-learn.org/stable/modules/feature_extraction.html, 4.2.3.7)

K = [1,5,10,20,40,60,80,100,500,1000] # Different values to be used in recall rate calculation
final_ans = [0 for x in range(0,len(K))] #Answer for different values of K, follows same index as above list
count_K = {} #Dictionary to map b/w K value(Used in RR@K) and which index it's located at
count = 0
for i in K:
	count_K[i] = count #count_K = {1:0,5:1 ...}
	count = count + 1
bid_title = {} #Dictionary between a bug's bug_id and its title char n gram (from 4 to 10), stored using frozenset 
bid_desc = {} # Same as above, only for the description
bid_mid = {} #Mapping b/w a bug's id and it's master id
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 10), min_df=1, lowercase=False,stop_words=None)
#To convert a word to collection of n grams, n going from 4 to 10

dups = open('DupsTest5000.csv')
dups_list = dups.readlines()
dups_list.pop(0) #First element is the pattern header
dups.close()
count_relevant_queries = 0 #Count of number of relevant bug queries
data_pattern = [('panda_id', int),('bug_id' ,int),('bug_title',str),('bug_desc',str),('master_id1',int),('master_id2',int),('master_title',str),('master_desc',str),('eof',str)] #pattern of a single entry

for report in xrange(0,len(dups_list)):
	portion = dups_list[report].split('\t') #portion[0] is panda_id, portion[1] is title of bug report, 
#portion[2] is title bug  report,  portion[3] is bug description, portion[4] is master bug id, 
#portion[5] == portion[4], portion[6] is master bug title, #portion[7] is master bug description, portion[8] is just eol stuff 
	print report
	portion[0] = int(portion[0])
	portion[1] = int(portion[1])
	for k in range(0,len(portion)-1):
		if(portion[k] == portion[k+1] and portion[k].isdigit()):
			portion[k] = int(portion[k])
			portion[k+1] = int(portion[k+1])
			break
	
	index2 = k #Stores the location where the second set of integers appear
	assert(len(portion)==9) # As per specification, portion should always be of length 9
	
	bid_mid[portion[1]] = portion[5] # relates bug id to master id
	count_relevant_queries = count_relevant_queries + 1
	bid_mid[portion[5]] = portion[5] # To be ignored while testing

	bug_title = portion[2].split(' ')	
	if(bug_title == ['']):
		 bid_title[portion[1]] = frozenset([''])
	else:
		ngram_vectorizer.fit_transform(bug_title) #Store set of bug's title ngrams
		bid_title[portion[1]] = frozenset(ngram_vectorizer.get_feature_names())

	master_title = portion[6].split(' ')
	if(master_title ==['']):
		bid_title[portion[5]] = frozenset([''])
	else:
		ngram_vectorizer.fit_transform(master_title) #Store set of master's title ngrams
		bid_title[portion[5]] = frozenset(ngram_vectorizer.get_feature_names())

	bug_desc = portion[3].split(' ')
	if(bug_desc==['']):
		 bid_desc[portion[1]] = frozenset([''])
	else:
		ngram_vectorizer.fit_transform(bug_desc) #Store set of bug's description ngrams
		bid_desc[portion[1]] = frozenset(ngram_vectorizer.get_feature_names())

	master_desc = portion[7].split(' ')
	if(master_desc==['']):
		 bid_desc[portion[5]] = frozenset([''])
	else:
		ngram_vectorizer.fit_transform(master_desc) #Store set of master's description ngrams
		bid_desc[portion[5]] = frozenset(ngram_vectorizer.get_feature_names())   

#Now iterate over all the bugs to calculate Recall Rate
#print bid_title print bid_mid print bid_desc

for bID in bid_mid:
	print bID
	if(bid_mid[bID]==bID): #Don't use master_bug's in the database to calculate recall rate
		continue
	score = [] #Store scores for a query, for all the logs in db
	for bug in bid_mid:
		if(bug == bID): # Queries score with itself is always highest, ignore
		   continue

		ngram_match_tt = len(bid_title[bID].intersection(bid_title[bug])) #Count matches b/w bugx title and bugy title
		ngram_match_td = len(bid_title[bID].intersection(bid_desc[bug])) # Count matches b/w bugx title and bugy desc
		ngram_match_dt = len(bid_desc[bID].intersection(bid_title[bug])) #Count matches b/w bugx desc and bugy title

		ngram_match_total = ngram_match_tt + ngram_match_td + ngram_match_dt
		score.append((bug,ngram_match_total)) # insert score into the list

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
