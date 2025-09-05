import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from config import DATA_PATH, SVM_MODEL_PATH,BAYESIAN_MODEL_PATH, VECTORIZER_PATH, TEST_SIZE, RANDOM_STATE, MODELS,MAX_FEATURES, NGRAM_RANGE
from preprocessing import preprocess_text

def train_model():
    os.makedirs(os.path.dirname(SVM_MODEL_PATH), exist_ok=True)
    # Load dataset
    df = pd.read_csv(DATA_PATH, header = None)
    df.rename(columns = {
                0: 'rate',
                1: 'content'
            }, inplace= True)
    
    df['processed'] = df['content'].apply(preprocess_text)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(df['processed'])
    y = df['rate']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # SVM MODEL
    param_grid_svm = MODELS["svm"]["params"]
    grid_svm = GridSearchCV(SVC(class_weight='balanced'), param_grid_svm, cv=5)
    grid_svm.fit(X_train, y_train)
    
    svm_model = grid_svm.best_estimator_
    # Save SVM model
    with open(SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm_model, f)

    # BAYESIAN MODEL
    param_grid_bayes = MODELS["naive_bayes"]["params"]
    grid_bayes = GridSearchCV(MultinomialNB(), param_grid_bayes, cv=5)
    grid_bayes.fit(X_train, y_train)
    
    bayesian_model = grid_bayes.best_estimator_

    # Save Naive Bayes model
    with open(BAYESIAN_MODEL_PATH, "wb") as f:
        pickle.dump(bayesian_model, f)

    # Save vectorizer
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Models trained and saved.")

if __name__ == "__main__":
    train_model()
