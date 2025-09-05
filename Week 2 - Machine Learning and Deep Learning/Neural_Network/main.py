from train import train_model
from data_loader import process_new_document
import numpy as np

model, history, id_to_category = train_model()

def predict_new_document(text):
    padded = process_new_document(text)
    pred = model.predict(padded)
    pred_id = np.argmax(pred, axis=1)[0]
    return id_to_category[pred_id]

