from typing import List
import nltk
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
import pickle	# this is for saving and loading your trained classifiers.
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
import json
import os


nltk.download("wordnet")
nltk.download('stopwords')

stopWords = set(stopwords.words("english"))
	
def removeSingleAndEmptyChar(words: List[str]):
	singleCharacterPattern = r'^[\w]{1}$'
	emptyStringPattern = r'^$'
	for word in words:
		if re.search(singleCharacterPattern, word) or re.search(emptyStringPattern, word):
			words.remove(word)	
	
def lowercaseStemAndLemmatize(token, stemmer, lemmatizer):
	return lemmatizer.lemmatize(stemmer.stem(token.lower()))

def sanitizeTokens(tokens: List[str], stemmer, lemmatizer):
	return [lowercaseStemAndLemmatize(token, stemmer, lemmatizer) for token in tokens if token.isalpha() and token.lower() not in stopWords]
	

def preprocess(filename, assignedClass):				
	file = open(filename, 'r')			
	lines = file.read().splitlines()
	file.close()
	
	# Tokenize, stem, lemmatize
	pattern = r'[\s\t.,?!:;\\]+'
	documents = []
	stemmer = nltk.stem.PorterStemmer()
	lemmatizer = nltk.stem.WordNetLemmatizer()
	
	for lineNumber in range(0, len(lines), 2):
		headerWords = re.split(pattern, lines[lineNumber])
		textWords = re.split(pattern, lines[lineNumber + 1])
		removeSingleAndEmptyChar(headerWords)
		removeSingleAndEmptyChar(textWords)
		document = []
		document.extend(sanitizeTokens(headerWords, stemmer, lemmatizer))
		document.extend(sanitizeTokens(textWords, stemmer, lemmatizer))
		documents.append({word: True for word in document})
	
	return list(zip(documents, [assignedClass for document in documents]))

def checkIfFileExists(filename):
	return os.path.isfile(filename)

def create_megadoc(env):
	# Read and saves megadoc
	megadocFilename = "megadoc_" + env + ".txt"
	training_megadoc = []
	if not checkIfFileExists(megadocFilename):
		genres = ["philosophy","sports","mystery","religion","science","romance","horror","science-fiction"]
		training_documents = [genre + "_" + env + ".txt" for genre in genres]
		dirName = "task1/" + env + "/"
	
		for index, filename in enumerate(training_documents):
			training_megadoc.extend(preprocess(dirName + filename, genres[index]))

		with open(megadocFilename , 'w') as fout:
			json.dump(training_megadoc, fout)
	else:
		file = open(megadocFilename)
		training_megadoc = json.load(file)
		file.close()
	
	return training_megadoc

def extract_features(megadoc):		# megadoc can be either training_megadoc for training phase or test_megadoc for testing phase.
	print("extractFeatures")
	return megadoc
	####################################################################################################################
	#																												   #	
	#		TO DO: Select features and create feature-based representations of labeled documents.                      #
	#																												   #
	####################################################################################################################


def train(classifier, trainingSet):
	return classifier.train(trainingSet)
	

def test(classifier, testSet):
	yTrue = [label for (_, label) in testSet]
	yPred = [classifier.classify(features) for (features, _) in testSet]

	accuracy = accuracy_score(yTrue, yPred)
	print("Accuracy:", accuracy)	

	recall = recall_score(yTrue, yPred, average="micro")
	print("Recall:", recall)

	precision = precision_score(yTrue, yPred, average="micro")
	print("Precision:", precision)

	f1Score = f1_score(yTrue, yPred, average="micro")
	print("F1-Score:", f1Score)

	print("\Confusion Matrix:")
	print(confusion_matrix(yTrue, yPred))


def save_classifier(classifier, filename):	#filename should end with .pickle and type(filename)=string
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return
	
	
def load_classifier(filename):	#filename should end with .pickle and type(filename)=string
	if checkIfFileExists(filename):
		classifier_file = open(filename, "rb")
		classifier = pickle.load(classifier_file)
		classifier_file.close()
		return classifier
	else:
		return False
	
def getClassifier(classifierName, trainingSet, env):
	filename = classifierName + "_" + env
	if classifierName == "naive":
		naiveBayesClassifier = load_classifier(filename)
		if not naiveBayesClassifier:
			naiveBayesClassifier = nltk.NaiveBayesClassifier
			trainedNaiveBayesClassifier = train(naiveBayesClassifier, trainingSet)
			save_classifier(trainedNaiveBayesClassifier, filename)
			return trainedNaiveBayesClassifier
		else:
			return naiveBayesClassifier
	elif classifierName == "svc":
		svcClassifier = load_classifier(filename)
		if not svcClassifier:
			svcClassifier = nltk.classify.SklearnClassifier(SVC())
			train(svcClassifier, trainingSet)
			save_classifier(svcClassifier, filename)

		return svcClassifier	
	else:
		raise Exception("Unknown classifier name:" + classifierName)


if __name__ == "__main__":
	env = "train" # train or dev
	trainingSet = create_megadoc(env)
	testSet = create_megadoc("test")

	trainingSet = extract_features(trainingSet)
	testSet = extract_features(testSet)

	naiveClassifier = getClassifier("naive", trainingSet, env)
	svcClassifier = getClassifier("svc", trainingSet, env)

	print("Naive Bayes:")
	test(naiveClassifier, testSet)

	print("Support Vector Classifier:")
	test(svcClassifier, testSet)









