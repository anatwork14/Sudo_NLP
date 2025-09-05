from sklearn.feature_extraction.text import CountVectorizer
from config import *
from sklearn.feature_extraction.text import TfidfVectorizer



def build_vectorizer(ngram, docs, type):
    if (type != 'TF-IDF'):
        vec = CountVectorizer(ngram_range=ngram)
        vec.fit_transform(docs)

        # print(vec.vocabulary_)

        print(vec.get_feature_names_out()[2000:2200])
        return vec
    else:
        tfdif_vec = TfidfVectorizer()
        tfdif_vec.fit_transform(docs)

        print("TF-IDF vocabulary:", tfdif_vec.get_feature_names_out()[20000:20200])


        print("TF-IDF number of unique words:", len(tfdif_vec.get_feature_names_out()))
        return tfdif_vec

    