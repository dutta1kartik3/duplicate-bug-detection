dup_file = file("BugData_duplicates2.tsv")
train_file = file("BugData_train2.tsv")

def getCharNGrams(word,n=3):
	return [word[i:i+n] for i in range(len(word)-n+1)]

unique_words = set()
unique_trigrams = set()

total_words = 0
total_trigrams = 0

total_docs = 0

for line in dup_file:
	try:
		total_docs+=1
		line = line.strip().split("\t")
		title, desc = line[1], line[2]
		title = title.split(" ")
		desc = desc.split(" ")
		total_words +=  len(title) + len(desc)
		unique_words.update(title+desc)
		for word in title+desc:
			trigrams = getCharNGrams("#" + word + "#")
			unique_trigrams.update(trigrams)
			total_trigrams += len(trigrams)
	except IndexError:
		print line

for line in train_file:
	try:
		total_docs+=1
		line = line.strip().split("\t")
		title, desc = line[1], line[2]
		title = title.split(" ")
		desc = desc.split(" ")
		total_words +=  len(title) + len(desc)
		unique_words.update(title+desc)
		for word in title+desc:
			trigrams = getCharNGrams("#" + word + "#")
			unique_trigrams.update(trigrams)
			total_trigrams += len(trigrams)
	except IndexError:
		print line

stats_file = open("BugStats_From_CSV_Files2.txt","w+")
stats_file.write("Total Reports: "+str(total_docs));
stats_file.write("\nTotal Unigrams: "+str(total_words));
stats_file.write("\nUnique Unigrams: "+str(len(unique_words)));
stats_file.write("\nTotal Char-Trigrams: "+str(total_trigrams));
stats_file.write("\nUnique Char-Trigrams: "+str(len(unique_trigrams)));
stats_file.write("\nAverage Unigrams per Report(Title + Desc) "+str( total_words/float(total_docs)));
