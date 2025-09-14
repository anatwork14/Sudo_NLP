# 📝 Text Generation with LSTM

This project implements a **Text Generation Model** using **Deep Learning (LSTM)**.  
It supports Vietnamese text processing with [`underthesea`](https://github.com/undertheseanlp/underthesea) and trains an LSTM-based model to generate new sequences of text.

---

## ⚙️ Requirements

- **Python 3.10.x** (required for `underthesea` compatibility)
- PyTorch
- Other dependencies listed in `requirements.txt`

---

## 🚀 Setup & Installation

Clone the project and create a virtual environment with Python **3.10**:

```bash
# Create venv with Python 3.10
py -3.10 -m venv venv

# Activate environment
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt
bash```

````

---

## 📂 Data Source

SOURCE:

1. Download the dataset (provided as a `.zip`) .
2. Unzip it into the `/data` folder.

```
project-root/
│── data/
│   ├── your_text_files.txt
│   ├── ...
│── main.py
│── model.py
│── train.py
│── predict.py
```

---

## 🧠 Training & Usage

- When you run the project, it will check for an existing model file:
  `textgen_model.pth`

- If the file does **not** exist, the script will **train a new model** automatically using:

```python
Trainer.train()
```

- After training, the model is saved as `textgen_model.pth` for future use.

---

## ⚡ Notes on Performance

Generated results may sometimes be **incoherent or meaningless**. This can happen because:

1. **Insufficient training data** –
   Currently, the number of documents is limited (`num_doc` in `config.py`, default: `10`).

2. **Data diversity** –
   The dataset combines multiple documents from `/data`.
   Each document may have a different **writing style** and **vocabulary usage**, leading to mixed results during training.

---

## 📌 Example

Run the main script:

```bash
python main.py
```

Input a seed text when prompted:

```
Enter the text for next generation: mùa xuân
```

Example output:

```
✨ Generated: mùa xuân đẹp và đầy sức sống ...
```

---

## 🛠️ Project Structure

```
├── config.py         # Configuration (e.g., num_doc, hyperparameters)
├── dataset.py        # Dataset preprocessing
├── loader.py         # Data loader
├── main.py           # Entry point
├── model.py          # LSTM model definition
├── predict.py        # Text generation logic
├── train.py          # Training script
├── utils.py          # Helper functions
└── data/             # Dataset (unzipped here)
```

---

## 📖 Summary

- ✅ LSTM-based text generation model
- ✅ Pretrained embeddings with `underthesea` tokenizer
- ✅ Auto-trains if no saved model is found
- ⚠️ Quality depends heavily on dataset size & consistency
