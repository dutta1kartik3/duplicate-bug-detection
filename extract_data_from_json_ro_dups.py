from sys import argv
from ast import literal_eval as le
import sys
from nltk.util import ngrams
import re
import string
reload(sys)  

sys.setdefaultencoding('utf8')
import re

uescapes = re.compile(r'(?<!\\)\\u[0-9a-fA-F]{4}', re.UNICODE)
def uescape_decode(match): return match.group().decode('unicode_escape')


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
seen = {}
bug_data_train_file = open("BugData_train_wo_rep.tsv","w+")
bug_data_dup_file = open("BugData_dups_wo_rep.tsv","w+")
dup_n=0


for line in f:
	total_docs+=1
	line = line.strip().replace("\n","\t")
	bug = le(line)
	try:
		description = bug["description"].replace("\n","\t")
		description = uescapes.sub(uescape_decode, description)
		title = bug["short_desc"].lower().replace("\n"," ")
		description = description.split("\t")
		temp = []
		for i in description:
			if "Additional Details".lower() in i.lower():
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
		description = description.replace('/','')
		description = description.replace('\\','')
		description = description.replace(':','')
		exclude = set(string.punctuation)
		table = string.maketrans("","")
		#description = description.translate(table, string.punctuation)
		#description = unicode(description, errors='ignore')
		
		description = re.sub(' +',' ',description)
		description = description.lower()
		if bug["bug_id"] not in seen:
			seen[bug["bug_id"]]=1
			dup = bug["dup_id"]
			if dup == []:
				bug_data_train_file.write(str(bug["bug_id"])+"\t"+title+"\t"+description+"\n")

			else:
				bug_data_dup_file.write(str(bug["bug_id"])+"\t"+title+"\t"+description+ "\t" +str(dup) +"\n")
				dup_n=dup_n+1
		else:
			seen[bug["bug_id"]]+=1

	except KeyError:
		print "KeyError"
	except AttributeError:
		print line
		

print seen
print "Reports with Duplicates: ",dup_n
print "Total Reports: ",total_docs
