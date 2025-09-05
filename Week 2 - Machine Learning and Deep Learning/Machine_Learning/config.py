# File paths
DATA_PATH = "train.csv"

SVM_MODEL_PATH = "models/svm_model.pkl"
BAYESIAN_MODEL_PATH = "models/bayesian_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

# Training params
TEST_SIZE = 0.2
RANDOM_STATE = 30

# Vectorizer
MAX_FEATURES = 10000
NGRAM_RANGE=(1, 3)

# Model hyperparameters
MODELS = {
    "naive_bayes": {
        "type": "MultinomialNB",
        "params": {
            "alpha": [0.1, 0.5, 1.0, 2.0]
        }
    },
    "svm": {
        "type": "LinearSVC",
        "params": {
            "C": [0.1, 1.0, 10.0],    # regularization strength
            'kernel': ['linear', 'rbf']
        }
    }
}