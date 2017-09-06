from collections import Counter
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re
import string

def preprocess(train_file):
	dataset = open( train_file, "r" )
	all_words = []
	labels = []

	stopWords = set(stopwords.words('english'))
	
	for line in dataset:
	    words = line.lower().split()
	    label = words.pop(0)
	    labels.append(label)

	    words_temp = []
	    for word in words:
		    #mengubah menjadi e-mail address
		    word = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddress',word)
			#mengubah menjadi http address
		    word = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddress',word)
			#mengubah menjadi moneysymbol
		    word = re.sub(r'£|\$', 'moneysymbol',word)
			#mengubah menjadi phonenumber
		    word = re.sub(r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b','phonenumbr',word)
		    #mengubah menjadi number 
		    word = re.sub(r'\d+(\.\d+)?', 'numbr',word)
		    word = re.sub(r'[^\w\d\s]', ' ', word)
		    word = re.sub(r'\s+', ' ', word)
		    word = re.sub(r'^\s+|\s+?$', '',word)
		    word = word.translate(word.maketrans("","", string.punctuation))
		    if word.lower() not in stopWords and len(word) != 1:
		        words_temp.append(word.lower())

	    all_words.append(words_temp)
	dataset.close()

	for idw, words in enumerate(all_words):
		wnl = WordNetLemmatizer()
		ps = PorterStemmer()
		for idx, word in enumerate(words):
			#lemmatisasi
			all_words[idw][idx] = wnl.lemmatize(word)
			#stemming
			all_words[idw][idx] = ps.stem(word)

	return all_words, labels

def make_Dictionary(dataset, length):
	all_words = []
	for line in dataset:
		all_words += line

	dictionary = Counter(all_words)
	dictionary = dictionary.most_common(length)

	return dictionary

def extract_Features(dataset, dictionary):
	feature_matrix = np.zeros((len(dataset), len(dictionary)))
	for idxline, words in enumerate(dataset):
		for word in words:
			for idxdict, val in enumerate(dictionary):
				if val[0] == word:
					feature_matrix[idxline, idxdict] = words.count(word)
	return feature_matrix

data, label_data = preprocess("dataset")
dictionary = make_Dictionary(data, 3000)
feature_data = extract_Features(data, dictionary)

f = open('dict', 'w')
f.write(str(dictionary))
f.close()

g = open('hasil2', 'w')
g.write(str(data))
g.close()

model2 = LinearSVC()
scores = cross_val_score(model2, feature_data, label_data, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
