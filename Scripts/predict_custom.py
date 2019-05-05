import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import ast
import numpy as np
import os
import ast
import urllib.request
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os.path
from datetime import datetime
from collections import Counter

nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')

month = "january"

from sklearn.externals import joblib

filename = "../Models/{}/LR_model_{}.joblib".format(month.title(), month)
lr = joblib.load(filename)

import pickle
pickle_in = open("../Models/{}/word_frequency_{}.picle".format(month.title(), month),"rb")
words_frequency = pickle.load(pickle_in)
df = pd.read_csv("../Datasets/custom_dataset.csv")[['URL', 'Category', 'Language']]
df = df[df['URL'].notnull()]
df['Weight_model'] = ''
df['lr_normal'] = ''
df['lr_max'] = ''

toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
char_blacklist = list(chr(i) for i in range(32, 127) if i <= 64 or i >= 91 and i <= 96 or i >= 123)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
top = 2500
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
       'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
       'Accept-Encoding': 'none',
       'Accept-Language': 'en-US,en;q=0.8',
       'Connection': 'keep-alive'}
row_counter = 0
for row_id, row in df.iterrows():
    row_counter += 1
    try:
        url = row['URL']
        req = urllib.request.Request(url, headers=hdr)
        html = urlopen(req, timeout=15).read()
        # html = urlopen(url, timeout=15).read()
        soup = BeautifulSoup(html, "html.parser")
        [tag.decompose() for tag in soup("script")]
        [tag.decompose() for tag in soup("style")]
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk.lower() for chunk in chunks if chunk)
        tokens = nltk.word_tokenize(text)

        from collections import Counter
        counter = 0
        features_pred = np.zeros(top * len(words_frequency)).reshape(len(words_frequency), top)
        c = [word for word, word_count in Counter(tokens).most_common(top)]
        for category in words_frequency.keys():
            for word in c:
                if word in words_frequency[category]:
                    features_pred[counter][words_frequency[category].index(word)] = 1
            counter+=1

        category_weight = []
        for i in features_pred:
            weight_cof = np.where(i == 1)[0]
            weight_sum = 0
            for cof in weight_cof:
                weight_sum += top - cof
            category_weight.append(weight_sum)

        cat_index = category_weight.index(max(category_weight))
        category = list(words_frequency.keys())[cat_index]
        feature = features_pred[cat_index].reshape(-1, top)
        print("url: {} . {} / {}".format(url, row_counter, len(df)))
        print('Category: ', row['Category'])
        print("My model: ",category)
        prediction = lr.predict(feature)
        print("LR normal: ", list(words_frequency.keys())[int(prediction[0])])
        df.at[row_id, 'Weight_model'] = category
        df.at[row_id, 'lr_normal'] = list(words_frequency.keys())[int(prediction[0])]
    except:
        print("{} - Failed. {} / {}".format(row['URL'], row_counter, len(df)))
        continue

df = df[df['Weight_model'] != '']
model_acc = len(df[df['Weight_model'] == df['Category']]) / len(df) * 100
lr_acc = len(df[df['lr_normal'] == df['Category']]) / len(df) * 100
print("My model accuracy: {}% | {} / {}".format(model_acc, len(df[df['Weight_model'] == df['Category']]), len(df)))
print("Logistic regression accuracy: {}% | {} / {}".format(lr_acc, len(df[df['lr_normal'] == df['Category']]), len(df)))
