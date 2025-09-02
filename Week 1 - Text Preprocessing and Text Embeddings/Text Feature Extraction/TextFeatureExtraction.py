

import pandas as pd
import json

with open('/content/drive/MyDrive/SudoCode/Week 1 - Text Preprocessing/news_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


docs = [d["content"] for d in data]

# # Approach 1: CountVectorizer with Unigram

from sklearn.feature_extraction.text import CountVectorizer
vec_unigram = CountVectorizer()


vec_unigram.fit_transform(docs)

vec_unigram.vocabulary_

vec_unigram.get_feature_names_out()[2000:2200]

doc1 = vec_unigram.transform([docs[0]]).toarray()

print(sum([1 if x == 1 else 0 for x in doc1[0]]))

doc1.shape

len(vec_unigram.get_feature_names_out())

# # Approach 2: CountVectorizer with Bigram

bigram_vec = CountVectorizer(ngram_range=(2,2))

bigram_vec.fit_transform(docs)

bigram_vec.vocabulary_

bigram_vec.get_feature_names_out()[2000:2200]

len(vec_unigram.get_feature_names_out())

# # Approach 3: TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfdif_vec = TfidfVectorizer()
X_tfidf = tfdif_vec.fit_transform(docs)


print("TF-IDF vocabulary:", tfdif_vec.get_feature_names_out()[20000:20200])


print("TF-IDF number of unique words:", len(tfdif_vec.get_feature_names_out()))





