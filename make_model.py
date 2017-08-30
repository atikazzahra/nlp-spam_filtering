from collections import Counter
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

print (make_Dictionary("dataset"))