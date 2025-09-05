# Comparison of SVM and Bayesian Models for Text Classification

> **Objective**: Build a text classification and compare the performance of **Linear SVM** and **Naive Bayes (Multinomial/Complement)** on the same dataset.

---

## 1) Setup

- **Requirements**: Python ≥ 3.9
- **Key libraries**: `scikit-learn`, `pandas`, `numpy`, `underthesea` for Vietnamese tokenization, `matplotlib`
- Quick install:

```bash
pip install requirements.txt
```

---

## 2) Text Preprocessing

**Goal**: Normalize Vietnamese text and represent features using TF‑IDF.

1. **Unicode normalization** (NFC/NFKC) and lowercase.
2. **Noise removal**: strip URLs, emails, @mentions, hashtags, numbers, emojis, repeated chars, extra punctuation. Keep Vietnamese diacritics.
3. **Tokenization**: `underthesea.word_tokenize()`.
4. **Feature representation**: `TfidfVectorizer`
5. **Data splitting**: `train/test`.
6. **Class imbalance handling**: for SVM use `class_weight='balanced'`; for NB prefer **ComplementNB**.

---

## 3) Models & Hyperparameters

### Linear SVM

- `sklearn.svm.LinearSVC` (fast, works well with sparse TF‑IDF)
- Tune: `C` (e.g., `[0.1, 1, 10]`), `class_weight`, `kernel`

### Naive Bayes

- `MultinomialNB` or `ComplementNB`
- Tune: `alpha` (e.g., `[0.1, 0.5, 1.0, 2.0]`)

### Hyperparameter search

- Use `GridSearchCV` or `RandomizedSearchCV` with 5‑fold CV, scoring by `f1_macro`.

---

## 4) Evaluation Metrics

| Model                  | Accuracy | Macro F1 | Weighted F1 | Notes                          |
| ---------------------- | :------: | :------: | :---------: | ------------------------------ |
| Linear SVM             |   82 %   |   81 %   |    82 %     | Stable, generalizes well       |
| Complement Naive Bayes |   77 %   |   76 %   |    77 %     | Very fast, strong with n‑grams |

---

## 5) Conclusion

- **SVM** usually achieves **higher F1** with TF‑IDF but is slower to train.
- **Naive Bayes** trains **extremely fast**, works well with short texts and small datasets; **ComplementNB** is especially useful when classes are **imbalanced**.

---

## 6) Reproducibility Steps

1. Normalize & tokenize text (Section 2).
2. Train both models (SVM & NB) with same TF‑IDF features.
3. Hyperparameter tuning with 5‑fold CV.
4. Evaluate on the **same test set** and log results.
