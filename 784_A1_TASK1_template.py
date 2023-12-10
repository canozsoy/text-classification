from typing import List
import nltk
import pickle	# this is for saving and loading your trained classifiers.
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
import json
import os
from sklearn import metrics

nltk.download("wordnet")
nltk.download('stopwords')
	
def remove_single_and_emoty_char(words: List[str]):
	single_character_pattern = r'^[\w]{1}$'
	empty_string_pattern = r'^$'
	for word in words:
		if re.search(single_character_pattern, word) or re.search(empty_string_pattern, word):
			words.remove(word)	
	
def lowercase_stem_and_lemmatize(token: str, stemmer, lemmatizer):
	return lemmatizer.lemmatize(stemmer.stem(token.lower()))

def sanitize_tokens(tokens: List[str], stemmer, lemmatizer):
	return [lowercase_stem_and_lemmatize(token, stemmer, lemmatizer) for token in tokens if token.isalpha() and token.lower() not in stop_words]
	

def pre_process(filename, assigned_class):			
	file = open(filename, 'r')			
	lines = file.read().splitlines()
	file.close()
	
	# Tokenize, stem, lemmatize
	pattern = r'[\s\t.,?!:;\\]+'
	documents = []
	stemmer = nltk.stem.PorterStemmer()
	lemmatizer = nltk.stem.WordNetLemmatizer()
	
	for line_number in range(0, len(lines), 2):
		header_words = re.split(pattern, lines[line_number])
		text_words = re.split(pattern, lines[line_number + 1])
		remove_single_and_emoty_char(header_words)
		remove_single_and_emoty_char(text_words)
		document = []
		document.extend(sanitize_tokens(header_words, stemmer, lemmatizer))
		document.extend(sanitize_tokens(text_words, stemmer, lemmatizer))
		documents.append({word: True for word in document})
	
	return list(zip(documents, [assigned_class for document in documents]))

def check_if_file_exists(filename):
	return os.path.isfile(filename)

def create_megadoc(env):
	# Read and saves megadoc
	megadoc_filename = "megadoc_" + env + ".txt"
	training_megadoc = []
	if not check_if_file_exists(megadoc_filename):
		genres = ["philosophy","sports","mystery","religion","science","romance","horror","science-fiction"]
		training_documents = [genre + "_" + env + ".txt" for genre in genres]
		dir_name = "task1/" + env + "/"
	
		for index, filename in enumerate(training_documents):
			training_megadoc.extend(pre_process(dir_name + filename, genres[index]))

		with open(megadoc_filename , 'w') as fout:
			json.dump(training_megadoc, fout)
	else:
		file = open(megadoc_filename)
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


def train(classifier, training_set):
	return classifier.train(training_set)
	

def test(classifier, test_set):
	y_true = [label for (_, label) in test_set]
	y_pred = [classifier.classify(features) for (features, _) in test_set]

	accuracy = metrics.accuracy_score(y_true, y_pred)
	print("Accuracy:", accuracy)	

	recall = metrics.recall_score(y_true, y_pred, average="micro")
	print("Recall:", recall)

	precision = metrics.precision_score(y_true, y_pred, average="micro")
	print("Precision:", precision)

	f1Score = metrics.f1_score(y_true, y_pred, average="micro")
	print("F1-Score:", f1Score)

	print("\Confusion Matrix:")
	print(metrics.confusion_matrix(y_true, y_pred))


def save_classifier(classifier, filename: str):
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return
	
	
def load_classifier(filename: str):
	if check_if_file_exists(filename):
		classifier_file = open(filename, "rb")
		classifier = pickle.load(classifier_file)
		classifier_file.close()
		return classifier
	else:
		return False
	
def getClassifier(classifier_name: str, training_set, env: str):
	filename = classifier_name + "_" + env
	if classifier_name == "naive":
		naive_bayes_classifier = load_classifier(filename)
		if not naive_bayes_classifier:
			naive_bayes_classifier = nltk.NaiveBayesClassifier
			trained_naive_bayes_classifier = train(naive_bayes_classifier, training_set)
			save_classifier(trained_naive_bayes_classifier, filename)
			return trained_naive_bayes_classifier
		else:
			return trained_naive_bayes_classifier
	elif classifier_name == "svc":
		svc_classifier = load_classifier(filename)
		if not svc_classifier:
			svc_classifier = nltk.classify.SklearnClassifier(SVC())
			train(svc_classifier, training_set)
			save_classifier(svc_classifier, filename)

		return svc_classifier	
	else:
		raise Exception("Unknown classifier name:" + classifier_name)


if __name__ == "__main__":
	stop_words = set(stopwords.words("english"))
	env = "train" # train or dev
	training_set = create_megadoc(env)
	test_set = create_megadoc("test")

	training_set = extract_features(training_set)
	test_set = extract_features(test_set)

	naive_classifier = getClassifier("naive", training_set, env)
	svc_classifier = getClassifier("svc", training_set, env)

	print("Naive Bayes:")
	test(naive_classifier, test_set)

	print("Support Vector Classifier:")
	test(svc_classifier, test_set)









