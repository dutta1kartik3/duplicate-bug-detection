
# coding: utf-8

# In[ ]:


from gensim.models import Word2Vec
from sys import argv
from gensim.models import Doc2Vec
from random import shuffle
from io import open
from gensim.models.doc2vec import TaggedDocument as TG

# import and setup modules we'll be using in this notebook
import logging
import itertools

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore


# In[ ]:



EPOCHS = 21
MODEL_DIR = "/lustre/amar/office_models/"


# In[ ]:


def get_model(dimensions, name):

    skipgram = Word2Vec(sg=1, size=dimensions, negative=5, min_count=2, workers=50)
    cbow = Word2Vec(sg=0, size=dimensions, negative=5, min_count=2,workers=50)
    pvdm = Doc2Vec(dm=1, size=dimensions, negative=5, min_count=2, workers=50)
    dbow = Doc2Vec(dm=0, size=dimensions, negative=5, min_count=2, workers=50, dbow_words = 1)

    if (name == 'skipgram'):
        return skipgram
    if (name == 'cbow'):
        return cbow
    if (name == 'pvdm'):
        return pvdm
    if (name == 'dbow'):
        return dbow




def train_model(model, sentences, dimensions, algorithm):
    epoch = 0
    model.build_vocab(sentences)
    while(epoch < EPOCHS):
        shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs = 1)	
        model.save(MODEL_DIR + "/" + algorithm + "_model_dimensions_"+str(dimensions)+"_epoch_"+str(epoch)+".word2vec")
        epoch += 1
        
# In[ ]:


def build_vocab(filename):
    f = open(filename)
    sentences = []
    for line in f:
        doc = line.split()
        doc = filter(lambda x: len(x) > 2, doc)
        sentences.append(doc)
    return sentences

def build_vocab_doc(filename):
    sentences = []
    f = file(filename)
    for line in f:
        line = line.strip().split()
        doc = line
        doc = filter(lambda x: len(x) > 2, doc)
        doc = TG(words = doc, tags = ['tag'])
        sentences.append(doc)
    return sentences



algorithms = [argv[1]] 
dimensions_to_use = [50,100,150,200,250]
model = []
for algorithm in algorithms:
    for dimension in dimensions_to_use:
        model = get_model(dimension, algorithm)
        senteces = build_vocab_doc('OfficeTrainFile.csv')
        print len(senteces)
        model = train_model(model, senteces, dimension, algorithm)


# In[ ]:





# In[ ]:




