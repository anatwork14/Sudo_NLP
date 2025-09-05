# data_loader.py
import os
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import config

def load_data(base_path):
    data = []
    document_categories = [
        d for d in os.listdir(base_path) 
        if os.path.isdir(os.path.join(base_path, d))
    ]
    
    for category in tqdm(document_categories, desc="Loading categories"):
        category_path = os.path.join(base_path, category)
        for file_name in tqdm(os.listdir(category_path), desc=f"Loading {category}", leave=False):
            file_path = os.path.join(category_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-16') as f:
                    text = f.read().strip()
                token_document = word_tokenize(text)
                process_document = ' '.join(token_document)
                data.append({'content': process_document, 'category': category})
            except UnicodeError:
                continue
    
    df = pd.DataFrame(data)
    return df, document_categories

def preprocess_text(train_texts, test_texts, train_labels, test_labels):
    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_texts + test_texts)

    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    train_padded = pad_sequences(train_sequences, maxlen=config.max_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=config.max_length, padding='post', truncating='post')

    return train_padded, test_padded, np.array(train_labels), np.array(test_labels), tokenizer

def process_new_document(text):
    # Preprocess
    tokenizer = Tokenizer(num_words=config.vocab_size, oov_token='<OOV>')
    segmented_text = ' '.join(word_tokenize(text))
    sequence = tokenizer.texts_to_sequences([segmented_text])
    padded = pad_sequences(sequence, maxlen=config.max_length, padding='post', truncating='post')
    return padded
