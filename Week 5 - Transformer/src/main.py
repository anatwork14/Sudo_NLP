from predict import *

predictor = Predict()

while True:
    x = input("Enter the English sentence (or just press Enter to quit): ").strip()
    if x == "":
        break
    print(predictor.translate(x))
