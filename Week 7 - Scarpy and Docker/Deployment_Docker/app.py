import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Initialize FastAPI Application ---
app = FastAPI(
    title="NLP Sentiment Prediction API",
    description="A microservice for classifying text sentiment (Positive=1, Negative=0)."
)

# --- 2. Define Input Data Schema (Pydantic) ---
class TextRequest(BaseModel):
    """Schema for the incoming request body."""
    text: str
    
class PredictionResponse(BaseModel):
    """Schema for the outgoing response body."""
    sentiment: str
    prediction_score: float

# --- 3. Load Trained Model Assets ---
MODEL_PATH = 'model.pkl'
TOKENIZER_PATH = 'tokenizer.pkl'

# Global variables to hold the loaded assets
model = None
tokenizer = None

@app.on_event("startup")
async def load_model_assets():
    """Load the model and tokenizer when the API server starts up."""
    global model, tokenizer
    try:
        logging.info(f"Loading model from {MODEL_PATH}...")
        model = joblib.load(MODEL_PATH)
        
        logging.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
        tokenizer = joblib.load(TOKENIZER_PATH)
        
        logging.info("Model and tokenizer loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Required files not found. Ensure '{MODEL_PATH}' and '{TOKENIZER_PATH}' exist.")
        # Raise an exception to prevent the server from starting with missing files
        raise RuntimeError("Model files missing. Run 'python setup_files.py' first.")
    except Exception as e:
        logging.error(f"Error loading model assets: {e}")
        raise

# --- 4. Define API Endpoints ---

@app.get("/")
def read_root():
    """Basic health check endpoint."""
    return {"status": "ok", "message": "NLP Prediction API is running."}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: TextRequest):
    """
    Accepts a text string and returns the predicted sentiment.
    """
    if not model or not tokenizer:
        raise HTTPException(status_code=503, detail="Model service not ready. Please try again in a moment.")

    try:
        # 1. Preprocess: Use the loaded tokenizer to transform the input text
        text_vector = tokenizer.transform([request.text])
        
        # 2. Predict: Get the probability scores (needed to determine the positive score)
        probabilities = model.predict_proba(text_vector)[0]
        positive_score = probabilities[1] # Assuming positive class is index 1

        # 3. Classify: Get the hard prediction (0 or 1)
        prediction_int = model.predict(text_vector)[0]
        
        # 4. Interpret: Map the integer prediction to a human-readable label
        sentiment_label = "Positive" if prediction_int == 1 else "Negative"
        
        return PredictionResponse(
            sentiment=sentiment_label,
            prediction_score=positive_score # Score towards the positive sentiment
        )

    except Exception as e:
        logging.error(f"Prediction failed for input: '{request.text}'. Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")

# --- 5. Uvicorn Server Command ---
# uvicorn app:app --reload --host 0.0.0.0 --port 8000