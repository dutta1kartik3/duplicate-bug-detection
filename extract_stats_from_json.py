from sys import argv
from ast import literal_eval as le

from nltk.util import ngrams
import re
import string




def getCharNGrams(word,n=3):
	return [word[i:i+n] for i in range(len(word)-n+1)]


def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


f = open(argv[1])
bi_grams = []
unigrams = 0
total_docs = 0
words = []

for line in f:
	total_docs+=1
	line = line.strip().replace("\n","\t")
	bug = le(line)
	try:
		description = bug["description"]
		title = bug["short_desc"]
		description = description.split("\t")
		temp = []
		for i in description:
			if i == "Additional Details :\n":
				print "continue"
				continue
			elif "Updated by" in i:
				if findnth(i,",",2)>-1:
					i = i[findnth(i,",",2)+1:]
				else:
					continue		
			elif "Created by" in i:
				if findnth(i,",",2)>-1:
					i = i[findnth(i,",",2)+1:]
				else:
					continue
			temp.append(i)
		description = " ".join(temp)	
		description = description.replace('"','')
		exclude = set(string.punctuation)
		table = string.maketrans("","")
		description = description.translate(table, string.punctuation)
		description = unicode(description, errors='ignore')
		description = re.sub(' +',' ',description)
		description = description.lower()
		test = 0
		tokens = description.split(" ")
		bi_grams += list(ngrams(tokens,2))
		unigrams += len(tokens)
		words += tokens
		tokens = title.split(" ")
		bi_grams += list(ngrams(tokens,2))
		unigrams += len(tokens)
		words += tokens

	except KeyError:
		print "KeyError"
	except AttributeError:
		print line
		


result_file = open("BugAnalysisJSON.txt", "w+")
result_file.write("Number of Docs:"+str(total_docs)+"\n")

result_file.write("Average Length of Doc:"+str(float(len(words))/total_docs)+"\n")

result_file.write("Number of Unigrams "+str(len(words))+"\n")

number_of_words = len(set(words))

result_file.write("Number of Unique Unigrams "+str(number_of_words)+"\n")

result_file.write("Number of Bigrams "+str(len(bi_grams))+"\n")

result_file.write("Number of Unique Bigrams "+str(len(set(bi_grams)))+"\n")

char_tri_grams = {}

unique_words = list(set(words))

for word in unique_words:
	tri_grams = getCharNGrams("#"+word+"#")
	for tri_gram in tri_grams:
		if tri_gram in char_tri_grams:
			char_tri_grams[tri_gram]+=1
		else:
			char_tri_grams[tri_gram]=1


	

result_file.write("Number of Unique Character Tri-grams "+str(len(char_tri_grams))+"\n")

number_of_tri_grams = sum(char_tri_grams.values())

result_file.write("Number of  Character Tri-grams "+str(number_of_tri_grams)+"\n")

result_file.close()

#print char_tri_grams



