import re
import math
import json
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from numpy.linalg import norm
from collections import Counter

import treetaggerwrapper

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from pprint import pprint as pp

import networkx as nx
import scipy.spatial as sp

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, TfidfModel
from gensim.models.ldamodel import LdaModel

class LdaModel:
	def __init__(self, input, output, limit):
		self.tagger = treetaggerwrapper.TreeTagger(TAGLANG='en')
		self.inp = input
		self.outp = output
		with open(self.inp + 'news_stops') as f:
			self.stopwords = f.read().strip().split()	
		self.excltags = ['CC', 'CD', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'RP', 'SYM', 'TO', 'WDT', 'WP', 'WP$', 'WRB', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP', 'VD', 'VDD', 'VDG', 'VDN', 'VDZ', 'VDP', 'VH', 'VHD', 'VHG', 'VHN', 'VHZ', 'VHP', 'RB', 'RBR', 'RBS', 'JJ', 'JJR', 'JJS']
		
	def load_data(corpusfile):
		corpus = pd.read_json(self.inp + corpusfile)
		articles = corpus.article_stripped.values
		return articles
	
	def strip_and_split(self, line): 
#remove whitespaces, split and lower text
		text_stripped = re.sub(r'[^\w\s]','',line).split()
		stop_words = self.stopwords#stopwords.words('english')
		text_stripped_= [word.lower() for word in text_stripped if not word.lower() in stop_words]
		return text_stripped_
	
	def make_bigram_model(self, words, ngram_mincount=20, threshold=50):
		bigram = gensim.models.Phrases(words, min_count=ngram_mincount, threshold=threshold)
		bigram_mod = gensim.models.phrases.Phraser(bigram)
		return bigram_mod
	
	def lemmatization(self, texts):
		texts_out = []
		tagged_texts = []
		for text in texts:
			tagged = self.tagger.tag_text(' '.join(text))
			tags = [tag.split('\t') for tag in tagged]
			rel = [t[2] for t in tags if t[1] not in self.excltags]
			texts_out.append(rel)
			tagged_texts.append(tags)
		return texts_out, tagged_texts
	
	def filter_most_common(self, corpus, limit=1000000):
		filtered_corpus = []
		ranked_vocab = Counter([f for l in data_lemmatized for f in l]).most_common(limit)
		filtered_vocab = [tup[0] for tup in ranked_vocab]
		for sent in corpus:
			filtered_sent = [word for word in sent if word in filtered_vocab]
			filtered_corpus.append(filtered_sent)
		return filtered_corpus, filtered_vocab
	
	def preprocess(self, corpusfile):
		articles = self.load_data(corpusfile)
		data_words = [self.strip_and_split(p) for p in articles]
		bigram_mod = self.make_bigram_model(data_words)
		with_bigrams = [bigram_mod[doc] for doc in data_words]
		data_lemmatized, tagged_texts = self.lemmatization(with_bigrams)
		data_filtered, filtered_vocab = self.filter_most_common(data_lemmatized, limit=10000)
		id2word = corpora.Dictionary(data_filtered)
		corpus = [id2word.doc2bow(text) for text in data_filtered]
		return data_filtered, id2word, corpus
		
	def make_model(self, texts, dictionary, corpus, num_topics, chunksize=1000, iterations=400, passes=40):
		model = LdaModel(corpus=corpus,
					id2word=id2word,
 					num_topics=num_topics,
				 	random_state=0,
					chunksize=chunksize,
 					iterations=iterations,
				 	passes=passes,
				 	alpha='asymmetric',
				 	per_word_topics=True)
		coherencemodel = CoherenceModel(model=model, texts=data_filtered, dictionary=id2word, coherence='c_v')
		coh = coherencemodel.get_coherence()
		topic_matrix = model.get_topics()
		return model, coh
	
	def compute_coherence_values(self, texts, dictionary, corpus, limit=8, start=3, step=1):
		coherence_values = {}
		models = []
		for num_topics in range(start, limit, step):
			model = LdaModel(corpus=corpus,
 			id2word=dictionary,
 			num_topics=num_topics,
 			random_state=0,
 			chunksize=1000,
 			iterations=400,
 			passes=40,
 			update_every=2,
 			alpha='asymmetric',
			per_word_topics=True)
		coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
		coherence_values[num_topics] = (coherencemodel.get_coherence())
		models.append(model)
		return coherence_values, models
	
	def label_paragraphs():
		pass
		
	def find_similar_topics():
		pass
