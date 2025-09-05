import os
import pickle
from preprocessing import preprocess_text
from config import SVM_MODEL_PATH, BAYESIAN_MODEL_PATH, VECTORIZER_PATH
from train import train_model

# Lazy-load models/vectorizer
def load_or_train():
    if not (os.path.exists(SVM_MODEL_PATH) and os.path.exists(BAYESIAN_MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        print("⚠️ No models found, training now...")
        train_model()

    with open(SVM_MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    with open(BAYESIAN_MODEL_PATH, "rb") as f:
        bayesian_model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return svm_model, bayesian_model, vectorizer


def predict(text: str, model_choice="svm"):
    svm_model, bayesian_model, vectorizer = load_or_train()
    processed = preprocess_text(text)
    vec = vectorizer.transform([processed])

    if model_choice == "svm":
        return rate_to_label(svm_model.predict(vec)[0])
    else:
        return rate_to_label(bayesian_model.predict(vec)[0])

def rate_to_label(rate):
  if (rate == 0):
    return 'Tệ'
  elif (rate == 1):
    return 'Vừa'
  else:
    return 'Tốt'

if __name__ == "__main__":
    new_text = "Sản phẩm mới này cùi quá, mọi người không nên sử dụng !!!"
    print("Prediction:", rate_to_label(predict(new_text)))
