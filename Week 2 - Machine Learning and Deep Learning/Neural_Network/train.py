# train.py
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import config
from data_loader import load_data, preprocess_text
from model import build_model

def train_model():
    # Load training & test data
    train_df, categories = load_data(config.train_folder)
    test_df, _ = load_data(config.test_folder)

    category_to_id = {category: idx for idx, category in enumerate(categories)}
    id_to_category = {idx: category for category, idx in category_to_id.items()}
    train_df['category_id'] = train_df['category'].map(category_to_id)
    test_df['category_id'] = test_df['category'].map(category_to_id)    
    
    # Split into texts & labels
    train_texts = train_df['content'].values
    train_categories = train_df['category_id'].values
    test_texts = test_df['content'].values
    test_categories= test_df['category_id'].values

    # Preprocess (tokenize + pad)
    train_padded, test_padded, train_categories, test_categories, tokenizer = preprocess_text(
        train_texts, test_texts, train_categories, test_categories
    )

    # One-hot encode labels
    num_classes = len(categories)
    train_labels_cat = to_categorical(train_categories, num_classes=num_classes)
    test_labels_cat = to_categorical(test_categories, num_classes=num_classes)

    # Build model
    model = build_model(num_classes)
    
    # Train
    history = model.fit(
        train_padded, train_labels_cat,
        epochs=config.epoch_size,
        batch_size=config.batch_size,
        validation_split=config.training_validation_split,
    )

    # Evaluate
    loss, acc = model.evaluate(test_padded, test_labels_cat, verbose=2)
    print(f"Test Accuracy: {acc:.4f}")

    return model, history, id_to_category
