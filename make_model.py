from collections import Counter
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import re
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

	    # words_temp = []
	    # for word in words:
	    #     if word not in stopWords and len(word) != 1:
	    #         words_temp.append(word)

	    all_words.append(words)
	dataset.close()

	# for idw, words in enumerate(all_words):
	# 	ps = PorterStemmer()
	# 	for idx, word in enumerate(words):
	# 		all_words[idw][idx] = ps.stem(word)

	return all_words, labels

def preprocess_text(messy_string):
    assert(type(messy_string) == str)
    porter = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    cleaned = re.sub(r'\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', messy_string)
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
    return ' '.join(
        porter.stem(term) 
        for term in cleaned.split()
        if term not in set(stop_words)
    )

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

#print(preprocess_text('WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.'))
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