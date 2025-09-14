# from loader import * 

# data = Loader()

# print(data.corpus)
from train import Trainer
from predict import Predictor
from config import *
class Main():
    def predict():
        model, itos, stoi = Trainer.train()
        text = input("Enter the text for next generation: ")
        print(Predictor.generate_text(model, text, stoi, itos, next_words=next_words))
        
    
if __name__ == "__main__":
    Main.predict()    
    Trainer.train()