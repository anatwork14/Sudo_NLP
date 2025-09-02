f = open('/content/drive/MyDrive/SudoCode/Week 1 - Text Preprocessing/news_dataset.json',encoding = 'utf-8')

raw_data = f.read()

working_data = raw_data[:3166]

# # Approach 1: Preprocessing with specialized tokenization
# 

nltk.download('punkt_tab')

import nltk, re
from nltk import word_tokenize

# **Objective**: Do thấy data ở dạng raw và có cấu trúc gồm cái field như id, author, content nên em có phương hướng xử lý chung như sau
# 
# Overall:
# 
# 1. Xử lý những whitespace character như \n\t, ... etc
# 2. Tokenize những thông tin đặc biệt đi theo cụm dễ hiểu nghĩa như `url`, thời gian timestamp không tính ngày tháng nghị quyết, email, đơn vị (300m, 12s, ...)
# 3. Xử lý `crawled_at` thành tokens theo kiểu ví dụ:
#    `"2022-08-01 09:08:13.106296" -> ['2022-08-01', '09:08:13.106296']`
# 

clean_re = re.sub(r'[\t\n\[\]\{\}\'\"]*', '', working_data).lower()

# Giữ lại website url, thời gian timestamp không tính ngày tháng nghị quyết, email, đơn vị (300m)
pattern = r'''(?x)     # set flag to allow verbose regexps
          (?:(?:\w+[ ]?\.[ ]?)+[\d]+)
          | (?:https?://[\w\d\.-]+(?:/[^\s]*)?|www\.[\w\d\.-]+) # Website url
          | (?:\d+[a-z]{1}\d*)                                  # Unit + thời gian như 1h30
          | (?:\d{2}:\d{2}:\d+.?\d+)                            # timestamp
          | [\w\.-]+@[\w\.-]+\.\w+                              # email
          | (?:\w+(?:[/-]\w+)+)                                 # slash/dash combos (2372/gp-stttt)
          | \w+                                                 # fallback: plain alphanumeric
          | \d+(?:[/-]\d+)+
'''
nltk.regexp_tokenize(clean_re, pattern)

clean_raw_data_nltk = nltk.wordpunct_tokenize(clean_re)

# # Approach 2: Preprocessing with normalize everything - treat it the same
# 

token_nltk = nltk.word_tokenize(working_data.lower())



# # Approach 3: Preprocessing in table - Sentence Segmentation
# 

import pandas as pd
data = pd.read_json('/content/drive/MyDrive/SudoCode/Week 1 - Text Preprocessing/news_dataset.json')

data["sentences"] = data["content"].apply(lambda x: nltk.sent_tokenize(str(x)))




