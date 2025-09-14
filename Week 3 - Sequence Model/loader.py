from config import folder, num_doc
from utils import *
import os 

class Loader:
    def __init__(self):
        self.corpus = []
        count = 0
        files = os.listdir(folder)
        for file in files:
            path = folder + "/" + file
            try:
                if (count == num_doc):
                    break
                self.corpus.append(Utils.read_file(path, "utf-8"))
                count +=1 
            except UnicodeDecodeError:
                self.corpus.append(Utils.read_file(path, "utf-16"))
                
    
