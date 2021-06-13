from sys import argv
import re
import string

f = file(argv[1])

def getCharNGrams(word,n=3):
	return [word[i:i+n] for i in range(len(word)-n+1)]


def findnth(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)

words = {}

for line in f:
	
	if line == "Additional Details :\n":
		print "continue"
		continue
	elif "Updated by" in line:
		if findnth(line,",",2)>-1:
			line = line[findnth(line,",",2)+1:]
		else:
			continue		
	elif "Created by" in line:
		if findnth(line,",",2)>-1:
			line = line[findnth(line,",",2)+1:]
		else:
			continue		
	#print line
	line = line.strip()
	line = line.replace('"','')
	exclude = set(string.punctuation)
	table = string.maketrans("","")
	line = line.translate(table, string.punctuation)
	line = line.decode('utf-8')	
	line = re.sub(' +',' ',line)
	line = line.lower()
	line = line.split(" ")
	#print len(line)
	#print line
	for word in line:
		if word in words:
			words[word]+=1
		else:
			words[word] = 1

result_file = open("Bug Analysis.txt", "w+")

result_file.write("Number of Unique Unigrams "+str(len(words))+"\n")

number_of_words = sum(words.values())

result_file.write("Number of Unigrams "+str(number_of_words)+"\n")

char_tri_grams = {}

for word in words.keys():
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


