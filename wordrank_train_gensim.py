from sys import argv
import time
from random import shuffle
from io import open
from gensim.models.word2vec import LineSentence as LS
from gensim.models.wrappers.wordrank import Wordrank
MODEL_DIR = "/lustre/amar/tokenized_Word2Vec_models"
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



time_file = open("TimeAnalysis.csv","a+")
def __main__():
	dimensions = int(argv[1])
	epochs = int(argv[2])
	
	
 	t = time.time()	

	model = Wordrank.train('./wordrank', 'WrappersTrainFile.txt',out_name=MODEL_DIR+'/wr_model'+str(epochs),size = dimensions, min_count = 2,iter = epochs, window = 5, loss = 'logistic')
	model.save(MODEL_DIR + "/Wordrank_model_dimensions_" + str(dimensions) + "_epoch_" + str(epochs)+".word2vec")
	time_file.write(unicode("WordRank,"+str(dimensions)+","+ str(EPOCHS) +"," + str(time.time() - t) + "\n"))
	time_file.close()
__main__()
