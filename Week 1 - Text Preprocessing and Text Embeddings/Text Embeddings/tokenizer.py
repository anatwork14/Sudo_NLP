from nltk import word_tokenize
import nltk
nltk.download('punkt_tab')

from gensim.models import Word2Vec
from config import *

def tokenize(data):
    tokens = word_tokenize(data)

    sentences = [tokens]

    model = Word2Vec(
        sentences,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        workers=WORKERS,
        epochs=EPOCHS,
        compute_loss=COMPUTE_LOSS 
    )

    model.build_vocab(sentences)
    model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=EPOCHS
    )
    print('✅ Load Model')
    return model
