"""
Data Extractor
@
@
@
"""

import json
import numpy as np
import cPickle as pickle
from nltk import word_tokenize
from collections import defaultdict as dd


def save_object(filename, obj):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)

class FormattedQuery():

	def __init__(self, data):
		self.query = data['query']
		self.passages = data['passages']
		self.query_id = data['query_id']
		self.query_type = data['query_type']

	def get_words(self):
		words = []
		query_words = word_tokenize(self.query.lower())
		words += query_words

		for passage in self.passages:
			words += word_tokenize(passage['passage_text'].lower())

		return words

	def transform(self, vocab, sentence_size):

		transformation = {}

		transformed_query = self.transform_query(vocab)
		transformed_passages = self.transform_passages(vocab, sentence_size)

		transformation['query'] = transformed_query
		transformation['passages'] = transformed_passages
		transformation['query_id'] = self.query_id
		transformation['query_type'] = self.query_type

		return transformation

	def transform_query(self, vocab):

		ref_query = 'STARTTOKEN ' + self.query.strip() + ' STOPTOKEN'
		ref_query = word_tokenize(ref_query)
		t_query = self.words_to_index(ref_query, vocab)

		return np.array(t_query)

	def transform_passages(self, vocab, sentence_size):

		passages_concat = []

		for passage in self.passages:
			p_toks = 'STARTTOKEN ' + passage['passage_text'].strip() + ' STOPTOKEN'
			p_toks = word_tokenize(p_toks)
			passages_concat += p_toks

		passages_indexes = self.words_to_index(passages_concat, vocab)
		passages_indexes = np.array(passages_indexes)
		division = len(passages_indexes)/sentence_size
		rest = len(passages_indexes)%sentence_size
		even, uneven = passages_indexes[:-rest], passages_indexes[-rest:]

		if len(even) == 0:
			chunks = [uneven]
		elif len(uneven) != 0:
			chunks = np.split(even, division) + [uneven]
		else:
			chunks = np.split(even, division)


		return chunks

	def words_to_index(self, words, vocab):

		list_of_indexes = []
		for word in words:
			if word in vocab:
				list_of_indexes.append(vocab[word])
			else:
				list_of_indexes.append(2)

		return list_of_indexes

def get_data(file_name):

	try:
		json_data = load_object('pkl/json_data.pkl')
	except:
		print 'Getting data...'
		file = open(file_name, 'r')
		lines = []
		for line in file:
			lines.append(line.strip())

		json_data = []
		for line in lines:
			json_data.append(json.loads(line))
		save_object('pkl/json_data.pkl', json_data)

	return json_data

def get_vocabulary(data, size):

	try:
		vocabulary = load_object('pkl/vocabulary.pkl')
	except:
		print 'Building vocabulary...'
		words = get_all_words(data)
		words_dict = dd(int)

		for word in words:
			words_dict[word] += 1

		word_items = sorted(words_dict.items(), key=lambda x: x[1], reverse=True)[:size]
		word_items = [i[0] for i in word_items]
		vocabulary = ['STARTTOKEN', 'STOPTOKEN', 'UNKTOKEN'] + word_items
		save_object('pkl/vocabulary.pkl', vocabulary)

	return {k: v for v, k in enumerate(vocabulary)}

def get_all_words(data):

	try:
		words = load_object('pkl/words.pkl')
	except:
		words = []
		print 'Collecting words...'
		for element in data:
			entry = FormattedQuery(element)
			words += entry.get_words()
		save_object('pkl/words.pkl', words)

	return words

def import_in_order(file_name, vocab_size=30000, sentence_size=15):
	
	try:
		 transformed_data = load_object('pkl/transformed_data.pkl')
	except:
		print 'Importing index arrays...'
		json_data = get_data(file_name)
		vocabulary = get_vocabulary(json_data, vocab_size)
		transformed_data = []

		for element in json_data:
			transformed_element = FormattedQuery(element).transform(vocabulary, sentence_size)
			transformed_data.append(transformed_element)
		save_object('pkl/transformed_data.pkl', transformed_data)

	return transformed_data






