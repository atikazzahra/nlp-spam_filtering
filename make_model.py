from collections import Counter
import numpy as np
import pandas as pd

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
	dictionary = dictionary.most_common(3000)

	return dictionary

def extract_Features(mail_dir, dic):
	dataset = open( mail_dir, "r" )
	len_dataset = (sum(1 for _ in dataset))

	feature_matrix = np.zeros((len_dataset, 3000))
	label_matrix = []
	for idxline, line in enumerate(dataset):
		words = line.split()
		label = words.pop(0)
		label_matrix += label
		for word in words:
			for idxdict, val in enumerate(dic):
				if val == word:
					feature_matrix[idxline, idxdict] = words.count(word)
	return feature_matrix, label_matrix


# dictionary = make_Dictionary("dataset")
# # feature_data, label_data = extract_Features("dataset", dictionary)

# # print(feature_data)
# # f = open('hasil', 'w')
# # np.set_printoptions(threshold=np.nan)
# # f.write(np.array_str(feature_data[0, :]))
# # f.close()
# # dataset = open( "dataset", "r" )
# # print (sum(1 for _ in dataset))
# # with dataset as ln:
# # 	print (sum(1 for _ in ln))

# dataset = open( "dataset", "r" )

# with 
# 	len_dataset = (sum(1 for _ in dataset))

# feature_matrix = np.zeros((len_dataset, 3000))
# label_matrix = []


# for idxline, line in enumerate(dataset):
# 	words = line.split()
# 	label = words.pop(0)
# 	label_matrix += label
# 	for word in words:
# 		print (word)
# 		for idxdict, val in enumerate(dictionary):
# 			if val == word:
# 				feature_matrix[idxline, idxdict] = words.count(word)

dataset = pd.read_csv("dataset", sep='/t', error_bad_lines=True)
print(dataset.head)
