from sys import argv
from unicodedata import category
f = file(argv[1])
clean_file = open("Train_Cleansed.txt","w+")


for s in f:

	s = ''.join(ch for ch in s.decode("utf-8") if category(ch)[0] != 'P' and ch not in ['+', '<', '>', '{', '}', '[',']'])
	s = s.encode("utf-8")
	temp = ""
	s = s.split("\t")
	for i in s[1:]:
		temp = temp + "\t" + " ".join(i.split())
	clean_file.write(temp[1:] + "\n")
	
clean_file.close()
