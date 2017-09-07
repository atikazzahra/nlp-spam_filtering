from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
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
	    line = preprocess_text(line)
	    words = line.lower().split()
	    label = words.pop(0)
	    labels.append(label)
	    all_words.append(words)
	dataset.close()

	return all_words, labels

def preprocess_text(text):
	assert(type(text) == str)
	wnl = WordNetLemmatizer()
	ps = PorterStemmer()
	stop_words = set(stopwords.words('english'))
	cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', text)
	cleaned = re.sub(r'(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr',
	                 cleaned)
	cleaned = re.sub(r'£|\$', 'moneysymb', cleaned)
	cleaned = re.sub(
	    r'\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
	    'phonenumbr', cleaned)
	cleaned = re.sub(r'\d+(\.\d+)?', 'numbr', cleaned)
	cleaned = re.sub(r'[^\w\d\s]', ' ', cleaned)
	cleaned = re.sub(r'\s+', ' ', cleaned)
	cleaned = re.sub(r'^\s+|\s+?$', '', cleaned.lower())
	cleaned = cleaned.translate(cleaned.maketrans("","", string.punctuation))
	cleaned = ' '.join(wnl.lemmatize(term) for term in cleaned.split() if term not in set(stop_words))
	return ' '.join(ps.stem(term) for term in cleaned.split())

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

model2 = LinearSVC()
scores = cross_val_score(model2, feature_data, label_data, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

model2.fit(feature_data, label_data)

significant_params = pd.Series(
    model2.coef_.ravel(),
    index= [x[0] for x in dictionary]
).sort_values(ascending=False)[:20]

for ind, val in significant_params.iteritems():
    print(ind,' | ',val)