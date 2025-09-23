from predict import *

predictor = Predict()

# Read the entire contents of input/input.txt
with open("input/input.md", "r", encoding="utf-8") as f:
    x = f.read()

# Generate and print the title
print(predictor.generate_title_lstm(x, max_len=15))
