from sys import argv
from gensim.models import Word2Vec
from random import shuffle


import time

t0 = time.time()

dimensions = int(argv[1])
num_epochs = int(argv[2])

#dup_file = file("BugData_duplicates.tsv")
#non_dup_file = file("BugData_train.tsv")
non_dup_file = file("BugData_train_wo_rep.tsv")
dup_file = file("DupTrainFile.tsv")
sentences = []

for line in non_dup_file:
	try:
		bud_id, title, desc = line.strip().split("\t")
		sentences.append(title.split(" ")+desc.split(" "))
	except ValueError:
		try:
			bug_id, content = line.strip().split("\t")
			sentences.append( content.split(" "))
		except ValueError:
			print line
for line in dup_file:
	try:
		bud_id, title, desc, dupId = line.strip().split("\t")
		sentences.append(title.split(" ")+desc.split(" "))
	except:
		print line

model = Word2Vec(sg=1, size=dimensions, negative=5, min_count=2,workers=1000)

model.build_vocab(sentences)

for epoch in range(0,num_epochs+1):	
	#shuffle(sentences)	
	model.train(sentences)	
	shuffle(sentences)
	if epoch%4 == 0:
		model.save("/lustre/amar/Word2Vec_models/Word2Vec_model_dimensions_"+str(dimensions)+"algo_1_epoch_"+str(epoch)+".word2vec")
		

    
non_dup_file.close()
dup_file.close()

t1 = time.time()

total = t1-t0
print "Time Taken", total
