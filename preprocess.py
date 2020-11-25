from gensim import corpora
import os
import numpy as np
import string
import codecs
import io
import gensim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection  import train_test_split
import pickle
from sklearn import neighbors
import re
from operator import itemgetter
from sklearn import metrics
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import GridSearchCV
from sklearn.model_selection  import RandomizedSearchCV
from time import time
from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics

from collections import namedtuple
from collections import Counter
import multiprocessing
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import spacy
import time

from sklearn import metrics
import sklearn
import pickle
from scipy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

import math
import random

from collections import defaultdict

class TS_SS:
	
	def Cosine(self, vec1: np.ndarray, vec2: np.ndarray):
		return np.dot(vec1, vec2.T)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))

	def VectorSize(self, vec: np.ndarray):
		return np.linalg.norm(vec)

	def Euclidean(self, vec1: np.ndarray, vec2: np.ndarray):
		return np.linalg.norm(vec1-vec2)

	def Theta(self, vec1: np.ndarray, vec2: np.ndarray):
		return np.arccos(self.Cosine(vec1, vec2)) + np.radians(10)

	def Triangle(self, vec1: np.ndarray, vec2: np.ndarray):
		theta = np.radians(self.Theta(vec1, vec2))
		return (self.VectorSize(vec1) * self.VectorSize(vec2) * np.sin(theta))/2

	def Magnitude_Difference(self, vec1: np.ndarray, vec2: np.ndarray):
		return abs(self.VectorSize(vec1) - self.VectorSize(vec2))

	def Sector(self, vec1: np.ndarray, vec2: np.ndarray):
		ED = self.Euclidean(vec1, vec2)
		MD = self.Magnitude_Difference(vec1, vec2)
		theta = self.Theta(vec1, vec2)
		return math.pi * (ED + MD)**2 * theta/360


	def __call__(self, vec1: np.ndarray, vec2: np.ndarray):
		return self.Triangle(vec1, vec2) * self.Sector(vec1, vec2)


def isAlphanumeric(text):
	return text.isalnum()

def tokenizeFile(filepath):
	# with open(filepath) as f:
	with codecs.open(filepath, encoding='utf-8',errors='ignore') as f:
		words = [(re.sub('[^a-zA-Z\']', ' ', line)).lower().split() for line in f]

		#print words
	#words.flatten()
	# words = [val for sublist in words for val in sublist]
	words = filter(isAlphanumeric,[val for sublist in words for val in sublist])
	# words = [val.decode("iso-8859-1") for sublist in words for val in sublist]

	return list(words)

def removeStopWords(wordDict, stopWords):
	toRemove = []
	for (i,w) in wordDict.items():
		if w in stopWords:
			toRemove.append(i)
	wordDict.filter_tokens(bad_ids=toRemove)
	wordDict.compactify()

def removeWordswithFreq_lessThanK(wordDict,freq,K):
	words = [k for k,v in freq.items() if v <= K]
	# print(words)
	# removeStopWords(wordDict,words)

#this one can load all data that is sorted by folder
def loadDataSet(path):
	counter = 1;
	filesAll = []
	labelsAll = np.zeros(0)
	for dirName in filter(os.path.isdir, [path + "/" + p for p in os.listdir(path)]):
		#print dirName
		files = [dirName + "/" +filename for filename in os.listdir(dirName)]
		labels = np.zeros(len(files))
		labels[:] = counter
		filesAll.extend(files)
		labelsAll = np.append(labelsAll,labels)
		counter += 1
	return filesAll,labelsAll

#load data with multiple dataset
def loadDataWithPreDefTest(pathTrain,pathTest):
	fileTrain, labelTrain = loadDataSet(pathTrain)
	fileTest, labeltest = loadDataSet(pathTest)

	return fileTrain, labelTrain, fileTest, labeltest


#return the train and test concateneted, and the index where the test set starts
def loadDataWithTest(pathTrain,pathTest):
	fileTrain, labelTrain = loadDataSet(pathTrain)
	fileTest, labelTest = loadDataSet(pathTest)
	files = fileTrain + fileTest
	labels = np.append(labelTrain,labelTest)
	#print len(labels)
	return files, labels, len(fileTrain)

def constructAndSave(filelist,is_TSSS,fname):

	cdict = constructDict(filelist)
	print(cdict)
	computeWordToWordMatrix(cdict,is_TSSS,fname)

def constructDict(fileList):
	ps = PorterStemmer()
	l = WordNetLemmatizer()
	freq = defaultdict(int)
	wordDict = corpora.dictionary.Dictionary()
	for file in fileList:
		doc = tokenizeFile(file)
		for x in doc:
			freq[ps.stem(x)]+=1
		wordDict.add_documents([[ps.stem(x)for x in doc]])

	print(len(freq))

	#removing stop word list
	removeStopWords(wordDict,list(stopwords.words('english')))

	removeWordswithFreq_lessThanK(wordDict,freq,5)
	return wordDict

def constructCorpus(fileList,ourDict):
	corpus = [ourDict.doc2bow(tokenizeFile(text)) for text in fileList]
	return corpus

def computeWordToWordMatrix(myDict,is_TSSS, fname):
	TS_SS_object = TS_SS()
	wordVectorModel = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
	wordToWord = np.eye((len(myDict)))
	for i in np.arange(len(myDict)): #(i,w) in myDict.items():
		try:
			#wx=wordVectorModel[myDict[i]]
			for j in np.arange(i): #(ii,ww) in myDict.items():
				try:
					#wwx=wordVectorModel[myDict[j]]
					if(is_TSSS):
						wordToWord[i,j] = TS_SS_object(wordVectorModel[myDict[i]] , wordVectorModel[myDict[j]])
					else:
						wordToWord[i,j] =  wordVectorModel.similarity(myDict[i],myDict[j])#np.dot(wx,wwx)
					
				except KeyError:
					None
					#nothing
		except KeyError:
			wordToWord[i,i] = 1
		print (i)

	a_tril = np.tril(wordToWord, k=0)
	a_diag = np.diag(np.diag(wordToWord))
	wordToWord = a_tril + a_tril.T - a_diag

	if(is_TSSS):
		fname = fname+"_TSSS"
	else:
		fname = fname + "_Cosine"

	np.save("preprocessedFiles/"+fname,wordToWord)
	with open("preprocessedFiles/"+fname+ '.pickle', 'wb') as handle:
		pickle.dump(myDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return wordToWord

def parallel_shuffle(A,B,seed = 1):
	temp = list(zip(A,B))
	random.Random(seed).shuffle(temp)
	A,B = zip(*temp)
	return A,B

def orderData(filepath,cc):
	#define
	testSize = 0.2
	if type(filepath) is tuple:
		(pathTrain,pathTest) = filepath
		files,labels,index = loadDataWithTest(pathTrain,pathTest)
		files = np.array(files)

		cc = 5
	else:
		files,labels = loadDataSet(filepath)
		# print ("load", filepath + "/" + str(cc) + 'perm.pickle')
		# with open(filepath + "/" + str(cc) + 'perm.pickle', 'rb') as handle:
		#        currentPerm = pickle.load(handle)
		#        currentPerm.astype(int)
		files = np.array(files)
		# files = files[currentPerm]
		# labels = labels[currentPerm]

		files,labels = parallel_shuffle(files,labels,cc)
		index = int(np.floor(len(labels)*(1-testSize)))
		cc = cc + 1
	trainFiles = files[0:index]
	testFiles = files[index:]
	trainLabels = labels[0:index]
	testLabels = labels[index:]
	return trainFiles, testFiles, trainLabels, testLabels, cc



if __name__ == '__main__':

	#datasetname : filepath 
	datasets = {

	# "bbcsport":"bbcsport",
	"twitter":"twitter",
	# "amazon":"amazon",
	# "20news":("20news/20news-bydate-train","20news/20news-bydate-test"),

	}


	for dataset_name,path in datasets.items():
		filepath = "datasets/"+path
		trainFiles, testFiles, trainLabels, testLabels, cc = orderData(filepath,0)
		total_files = trainFiles + testFiles
		print("Contructing Vocab and W2W for ", dataset_name)

		constructAndSave(total_files,False,dataset_name)
		constructAndSave(total_files,True,dataset_name)
