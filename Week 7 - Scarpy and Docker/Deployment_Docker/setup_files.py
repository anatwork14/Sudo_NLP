import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- 1. Train and Save Mock TfidfVectorizer (Tokenizer) ---
logging.info("Creating and training mock TfidfVectorizer (tokenizer)...")
corpus = [
    "great movie, loved the acting", 
    "terrible script and boring plot",
    "highly recommended, fantastic work",
    "worst experience, don't watch this"
]
# Initialize and fit the vectorizer
tokenizer = TfidfVectorizer()
tokenizer.fit(corpus)

# Save the tokenizer object
try:
    joblib.dump(tokenizer, 'tokenizer.pkl')
    logging.info("Successfully saved tokenizer.pkl")
except Exception as e:
    logging.error(f"Error saving tokenizer.pkl: {e}")

# --- 2. Train and Save Mock Classification Model ---
logging.info("Creating and training mock LogisticRegression model...")
X = tokenizer.transform(corpus)
# Simple mock labels (1=Positive, 0=Negative)
y = np.array([1, 0, 1, 0])

# Initialize and fit the model
model = LogisticRegression()
model.fit(X, y)

# Save the model object
try:
    joblib.dump(model, 'model.pkl')
    logging.info("Successfully saved model.pkl")
except Exception as e:
    logging.error(f"Error saving model.pkl: {e}")

logging.info("\nSetup complete. Run 'python app.py' to start the API.")