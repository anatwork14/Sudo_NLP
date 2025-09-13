import os
import json
folder = 'data'

data = os.listdir(folder)
print(data[:5])

author_dict = {}

for story in data:
    author = story.split("-")[1]
    author = author.split(".")[0]
    author = author.strip()
    if author not in author_dict:
        author_dict[author] = 1
    else :
        author_dict[author] += 1

final = dict(sorted(author_dict.items(), key=lambda item: item[1]))

with open('author.json', 'w', encoding='utf-8') as f:
    json.dump(final, f, ensure_ascii=False, indent=4)