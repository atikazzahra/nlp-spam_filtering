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

def preprocess(train_file):
	dataset = open( train_file, "r" )
	all_words = []
	labels = []

	stopWords = set(stopwords.words('english'))
	for line in dataset:
	    words = (line.lower()).split()
	    label = words.pop(0)
	    labels.append(label)

	    words_temp = []
	    for word in words:
	        if word not in stopWords and word.isalpha() == True and len(word) != 1:
		        words_temp.append(word)

	    all_words.append(words_temp)
	dataset.close()

	for idw, words in enumerate(all_words):
		ps = PorterStemmer()
		for idx, word in enumerate(words):
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


# data, label_data = make_Dictionary("dataset")
# feature_data = extract_Features("dataset", dictionary)

# #model1 = MultinomialNB()
# model2 = LinearSVC()
# #model1.fit(feature_data, label_data)
# model2.fit(feature_data, label_data)

# #model2 = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(model2, feature_data, label_data, cv=10)

# #feature_test, label_test = extract_Features("dataset", dictionary)
# #result1 = model1.predict(feature_test)
# #result2 = model2.predict(feature_test)
# #print (confusion_matrix(label_test,result1))
# #print (confusion_matrix(label_test,result2))
# print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
# f = open('dict', 'w')
# # np.set_printoptions(threshold=np.nan)
# f.write(str(dictionary))
# f.close()

data, label_data = preprocess("dataset")
dictionary = make_Dictionary(data, 3000)
feature_data = extract_Features(data, dictionary)

model2 = LinearSVC()
scores = cross_val_score(model2, feature_data, label_data, cv=10)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

f = open('dict', 'w')
# # np.set_printoptions(threshold=np.nan)
f.write(str(dictionary))
f.close()

# g = open('hasil', 'w')
# g.write(str(data))
# g.close()

# g = open('feature', 'w')
# g.write(np.array_str(feature_data[5574 : ]))
# g.close()
