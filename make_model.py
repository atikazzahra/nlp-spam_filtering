from collections import Counter
import numpy as np

def make_Dictionary(train_dir):
	dataset = open( train_dir, "r" )
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
	dictionary = dictionary.most_common(3000)

	return dictionary

def extract_Features(mail_dir):
	dataset = open( mail_dir, "r" )
	feature_matrix = np.zeros(len(dataset), 3000)
	
	for idxline, line in dataset:
		words = line.split()
		label = words.pop(0)
		for word in words:
			for idxdict, val in enumerate(dictionary):
				if val[0] == word:
					feature_matrix[idxline, idxdict] = words.count(word)
	return feature_matrix;


print (make_Dictionary("dataset"))