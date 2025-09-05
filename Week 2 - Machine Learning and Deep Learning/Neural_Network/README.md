# Text Classification — SVM vs Bayesian (Concise README)

## Quick start

1. **Create & activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate     # Windows (PowerShell)
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare data**
   Unzip the provided archives into the project root so the folders are:

```
Train_Full/
Test_Full/
```

---

## Project flow (short & clear)

1. **Preprocessing**

   - Unicode normalize, lowercase, remove noise (URLs, emails, extra punctuation).
   - Tokenize Vietnamese text (e.g. `underthesea.word_tokenize`).
   - Remove stopwords and normalize tokens.

2. **Feature extraction**

   - For neural baseline (notebook): `Tokenizer` → sequences → padded sequences → Embedding.

3. **Training**

   - Neural baseline: Keras `Embedding` + `Bidirectional(LSTM)` as in `demo.ipynb`.

---

## How to run

- **Notebook (LSTM baseline)**

```bash
jupyter notebook demo.ipynb
```

## Follow the cells.
