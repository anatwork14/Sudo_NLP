# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
import config

def build_model(num_classes):
    model = Sequential()
    model.add(Embedding(input_dim=config.vocab_size, output_dim=128, input_length=config.max_length))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(config.dropout_rate))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(config.dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=config.optimizer_algorithm, 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model
