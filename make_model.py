from collections import Counter
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def make_Dictionary(train_file):
	dataset = open( train_file, "r" )
	all_words = []
	for line in dataset:
	    words = line.split()
	    words.pop(0)
	    all_words += words
	dataset.close()

	dictionary = Counter(all_words)

	list_to_remove = list(dictionary)
	for item in list_to_remove:
	    if item.isalpha() == False:
	        del dictionary[item]
	    elif len(item) == 1:
	        del dictionary[item]

	stopWords = set(stopwords.words('english'))
	for item in list_to_remove:
	    if item in stopWords:
	        del dictionary[item]

	dictionary = dictionary.most_common(3000)

	return dictionary

def extract_Features(mail_dir, dic):
	dataset = open( mail_dir, "r" )
	len_dataset = (sum(1 for _ in dataset))

	dataset.seek(0)
	feature_matrix = np.zeros((len_dataset, 3000))
	label_matrix = []
	for idxline, line in enumerate(dataset):
		words = line.split()
		label = words.pop(0)
		label_matrix.append(label)
		for word in words:
			for idxdict, val in enumerate(dic):
				if val[0] == word:
					feature_matrix[idxline, idxdict] = words.count(word)
	dataset.close()
	return feature_matrix, label_matrix


dictionary = make_Dictionary("dataset")
feature_data, label_data = extract_Features("dataset", dictionary)

#model1 = MultinomialNB()
model2 = LinearSVC()
#model1.fit(feature_data, label_data)
model2.fit(feature_data, label_data)

#model2 = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(model2, feature_data, label_data, cv=10)

#feature_test, label_test = extract_Features("dataset", dictionary)
#result1 = model1.predict(feature_test)
#result2 = model2.predict(feature_test)
#print (confusion_matrix(label_test,result1))
#print (confusion_matrix(label_test,result2))
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
f = open('dict', 'w')
# np.set_printoptions(threshold=np.nan)
f.write(str(dictionary))
f.close()
