from underthesea import word_tokenize

def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join(tokens)
