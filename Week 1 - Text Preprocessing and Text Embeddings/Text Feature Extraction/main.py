from data_loader import *
from vectorizer import *
from config import * 

docs = loader(PATH)

# Unigram
vec_unigram = build_vectorizer(NGRAM_RANGE_UNIGRAM, docs, "unigram")

doc1 = vec_unigram.transform([docs[0]]).toarray()

print(sum([1 if x == 1 else 0 for x in doc1[0]]))

print(doc1.shape)

# Bigram
vec_bigram = build_vectorizer(NGRAM_RANGE_BIGRAM, docs, 'bigram')

doc2 = vec_bigram.transform([docs[0]]).toarray()

print(sum([1 if x == 1 else 0 for x in doc2[0]]))

print(doc2.shape)

# TF-IDF
tfidf_vec = build_vectorizer(None, docs, 'TF-IDF')
