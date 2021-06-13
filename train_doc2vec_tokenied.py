from sys import argv
from gensim.models import Doc2Vec
from random import shuffle
from gensim.models.doc2vec import TaggedDocument

dimensions = int(argv[1])
num_epochs = int(argv[2])
algo = int(argv[3])

train_file = file("Train_file_Word2Vec.txt")
sentences = []
count = 0
print "Reading Training File"
for line in train_file:
	line = line.strip().replace("\t"," ")
	for i in line:
		temp = TaggedDocument(words = i.split(" "), tags=[count] )
		sentences.append(temp)
	count+=1
print "Read Training File"
model = Word2Vec(dm=algo, size=dimensions, negative=5,  min_count=1,workers=10)

model.build_vocab(sentences)
print "vocab built"
for epoch in range(0,num_epochs+1):	
	shuffle(sentences)	
	model.train(sentences)	
	if epoch%4 == 0:
		model.save("/lustre/amar/tokenized_Word2Vec_models/Doc2Vec_model_dimensions_"+str(dimensions)+"algo_"+ str(algo) +"_epoch_"+str(epoch)+".doc2vec")
	print epoch
		

    

