import nltk
#import sklearn
import string
import pickle	# this is for saving and loading your trained classifiers.

#from sklearn.svm import SVC
from nltk.classify.scikitlearn import SklearnClassifier

##############################################################
#		If you need, you may import other packages etc.      #
##############################################################





### ***** YOU MAY ADD NEW FUNCTIONS, MODIFY OR DELETE EXISTING ONES. YOU MAY EVEN TOTALLY DISCARD THIS TEMPLATE AND CODE YOUR OWN SOLUTION FROM SCRATCH. ***** ###





def preprocess(filename):	#filename is of type string. example call: preprocess("philosophy_test.txt")	
	####################################################################################################################
	#		TO DO: Apply basic text processing steps which you think are required. (tokenizationi stemming etc.)       #
	#  			   If you want to do further preprocessing (e.g. removing number etc.), you can apply those here.      #
	#																												   #
	####################################################################################################################

	filepath = filename					# you might need to change filepath depending where do you store your data files.
	file = open(filepath, 'r')			# 'r' for read
	lines = file.read().splitlines()	# lines is a list holding each line of your file as strings. e.g. lines = ["This is the 1st line of the file", "This is the 2nd line of the file", ...]
	file.close()

	processed = []		# fill this list with the preprocessed form of the text in the file. you may change to another data structure, if you need. 
						# do not forget to label your documents. 
						# if a document from Class1 becomes "world case speak silent" after basic text processing steps; after labeling, it will look like ("world case speak silent", Class1)
	#...
	#...
	#...
	#...
	#...
	#...
	#...
	#...

	# At this point, *processed* should look like this: [(("world case speak silent", Class1), ("second docu come here", ClassN), ....., ("another docu continu finish", ClassX), ....]
	return processed 	# you may change the return value if you need.
	
	
	
	
	
	
	
######################################################################################################################################################
# This part is not compulsory. However, merging preprocessed forms of your files into mega documents may be pretty helpful.
# You may also want to permanently store (i.e. write to a file) those mega documents so as not to preprocess your file again and again at each trial.
def create_training_megadoc():
	training_documents = ["philosophy_train.txt","sports_train.txt","mystery_train.txt","religion_train.txt","science_train.txt","romance_train.txt","horror_train.txt","science-fiction_train.txt"]
	training_megadoc = []
	
	for filename in training_documents:
		training_megadoc += preprocess(filename)
	#####	
	#...
	# Here, you may write the training_megadoc to a file. (You may also do it elsewhere or nowhere.)
	#...
	#####
	return training_megadoc
	
	
def create_test_megadoc():
	training_documents = ["philosophy_test.txt","sports_test.txt","mystery_test.txt","religion_test.txt","science_test.txt","romance_test.txt","horror_test.txt","science-fiction_test.txt"]	
	test_megadoc = []
	#...
	#...
	#... *** TO DO ***
	#...
	#...
	return test_megadoc
		
####################################################################################################################################################






def extract_features(megadoc):		# megadoc can be either training_megadoc for training phase or test_megadoc for testing phase.
	####################################################################################################################
	#																												   #	
	#		TO DO: Select features and create feature-based representations of labeled documents.                      #
	#																												   #
	####################################################################################################################






def train(classifier, training_set):	# classifier is either nltk.NaiveBayesClassifier or SklearnClassifier(SVC()). Example call: train(SklearnClassifier(SVC()), training_set)
	####################################################################################################################
	#																												   #	
	#		TO DO: Use feature-based representations of your labeled documents to train your classifier.			   # 
	#																												   #
	####################################################################################################################

	
	

	

def test(classifier, test_set):
	####################################################################################################################
	#																												   #	
	#		TO DO: Use feature-based representations of your labeled documents to test your trained classifier.		   #
	#	 Compute accuracy, recall, precision values and confusion matrices. Present and discuss them at your report.   # 
	#																												   #
	####################################################################################################################






def save_classifier(classifier, filename):	#filename should end with .pickle and type(filename)=string
	with open(filename, "wb") as f:
		pickle.dump(classifier, f)
	return
	
	
def load_classifier(filename):	#filename should end with .pickle and type(filename)=string
	classifier_file = open(filename, "rb")
	classifier = pickle.load(classifier_file)
	classifier_file.close()
	return classifier




if __name__ == "__main__":
	# You may add or delete global variables.
	training_set = []
	test_set = []
	
	##############################
	#							 #
	#	 THE STAGE IS YOURS!	 #	
	#							 #
	##############################








