# -*- coding: utf-8 -*-

import nltk
import sklearn
import string
import random
import pickle
import numpy

import sklearn_crfsuite
from sklearn_crfsuite import scorers
# from sklearn_crfsuite import metrics
from sklearn import metrics

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer


def formatdata(formatted_sentences,formatted_labels,file_name):
	#file=open("en-ud-dev.conllu","r")
	file=open(file_name, 'r', encoding='ascii', errors='backslashreplace')
	#file=open(file_name,"rb")
	print("Reading data...")
	#quit()
	text=file.read().splitlines()
	tokens=[]
	labels=[]
	for line in text:
		line=line.split('\t')
		if len(line)==3:
			tokens.append(line[0])
			if line[1]=="PUNCT":
				labels.append(line[0]+"P")
			else:
				labels.append(line[2])	
		else:
			formatted_sentences.append(tokens)
			formatted_labels.append(labels)
			tokens=[]
			labels=[]



	

def creatdict(sentence,index,pos):	#pos=="" <-> featuresofword  else, relative pos (str) is pos
	word=sentence[index]
	wordlow=word.lower()
	dict={
		"wrd"+pos:wordlow,								# the token itself
		"cap"+pos:word[0].isupper(),					# starts with capital?
		"allcap"+pos:word.isupper(),					# is all capitals?
		"caps_inside"+pos:word==wordlow,				# has capitals inside?
		"nums?"+pos:any(i.isdigit() for i in word),		# has digits?
	}	
	return dict
	

def feature_extractor(sentence,index):
	features=creatdict(sentence,index,"")

	return features



			
def creatsets(file_name):	
	sentences=[]
	labels=[] 	#y_train (will be)
	formatdata(sentences,labels,file_name)	
	limit=int(len(sentences)/5)##############**********CHANGE these. these just limit the size of training set for faster trials. #####################
	sentences=sentences[:limit]##############
	labels=labels[:limit]####################
	
	#print(len(sentences),len(labels))			
	#print(formatted_sentences)
	#print(formatted_labels)
	print("Feature extraction...")
	features=[]		#X_train
	for i in range(0,len(sentences)):
		features.append([])
		for j in range(0,len(sentences[i])):
			features[-1].append(feature_extractor(sentences[i],j))
			
	del sentences[:]
	del sentences

	
	delimit=int((len(labels)*8)/10)
	test_data=[features[delimit:],labels[delimit:]]
	features=features[:delimit]
	labels=labels[:delimit]
	
	training_data=[features,labels]

	
	with open('pos_crf_train.data', 'wb') as file:
		pickle.dump(training_data, file)
	file.close()


	with open('pos_crf_test.data', 'wb') as file:
		pickle.dump(test_data, file)
	file.close()
		
	return training_data, test_data	
	
	
		
def train(training_data):		
	print("Training...")
	features=training_data[0]
	labels=training_data[1]	
	classifier.fit(features,labels)	
	



def test(test_data):
	print("Testing...")

	y_true=test_data[1]  #labels
	y_pred=classifier.predict(test_data[0])

	
	precision = metrics.precision_score(y_true, y_pred, average="micro")
	recall = metrics.recall_score(y_true, y_pred, average="micro")
	f1 = metrics.f1_score(y_true, y_pred, average="micro")
	accuracy = metrics.accuracy_score(y_true, y_pred)

	print("accuracy:",accuracy)
	print("f1:",f1)
	print("precision:",f1)
	print("recall:",recall)	

def save(filename):	#filename shall end with .pickle and type(filename)=string
	print("Saving classifier.")
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return
		
		
def load(filename):	#filename shall end with .pickle and type(filename)=string
	print("Loading classifier...")
	with open(filename, "rb") as f:
		classifier=pickle.load(f)
		return classifier

def flat_list(list):
	flat_list = []
	for row in list:
		flat_list.extend(row)
	
	return flat_list



if __name__ == "__main__":

	vectorizer = DictVectorizer()
	classifier=LogisticRegression(max_iter=1000) #sklearn_crfsuite.CRF(c1=0.2, c2=0.2, max_iterations=1000)
	training_data, test_data=creatsets("task2/en-ud-train.conllu")

	flat_features = flat_list(training_data[0])
	flat_labels = flat_list(training_data[1])
	vectorized_features = vectorizer.fit_transform(flat_features)
	
	
	with open('pos_crf_train.data', 'rb') as file:
		training_data=pickle.load(file)
	file.close()
	
	
	train([vectorized_features, flat_labels])
	#quit()
	save("pos_crf.pickle")
	
	
	with open('pos_crf_test.data', 'rb') as file:
		test_data=pickle.load(file)
	file.close()
	
	classifier=load("pos_crf.pickle")
	test([vectorizer.transform(flat_list(test_data[0])), flat_list(test_data[1])])
	
	s=['The',
	'guitarist',
	'died',
	'of',
	'a',
	'drugs',
	'overdose',
	'in',
	'1970',
	'aged',
	'27',
	'.']
