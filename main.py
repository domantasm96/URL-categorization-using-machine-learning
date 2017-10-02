import datetime
import csv
import nltk
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from langdetect import detect


# nltk.download('punk')
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

file = 'URL-categorization_10000.csv'
counter = 1
reader = csv.reader(open(file), delimiter=',')
header = next(reader)
char_blacklist = list(chr(i) for i in range(32, 127) if i <= 64 or i >= 91 and i <= 96 or i >= 123)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
language_whitelist = ['en']
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
top = 50
cv_number = 5
data = []
tokens_list = []
freq_words = []
filter_data = []
labels = []
print(datetime.datetime.now().time())

for row in reader:
    data.append(row)

for url_counter, row in enumerate(data):
    if row[5] != 'Not_working' and float(row[6]) > 0.5:
        try:
            url = 'http://' + row[-1]
            html = urlopen(url, timeout=1).read()
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk.lower() for chunk in chunks if chunk)
            text_vocab = set(w.lower() for w in text if w.lower().isalpha())
            if detect(text) in language_whitelist:
                continue
            tokens = nltk.word_tokenize(text)
            tokens_list += [nltk.word_tokenize(text)]
            print('----')
            print('URL: ' + row[-1])
            print('CATEGORY: ' + row[5])
            print(url_counter)
            print('Nr: ', counter)
            filter_data += [row[5]]
            counter += 1

        except:
            pass
print('Filtered English URL')
f1 = nltk.FreqDist(filter_data).most_common()
f2 = list(category for category, number in f1 if number >= cv_number)
all_categories = list(set(f2))
print('LABELS CREATION')
counter = 0
for index, word in enumerate(filter_data):
    if word in all_categories:
        labels += [all_categories.index(word)]
    else:
        tokens_list.pop(index - counter)
        counter += 1
save = labels
labels = np.array(labels).reshape((len(labels), 1))
print('CREATE FREQUENT WORDS LIST')
for tokens in tokens_list:
    allWordDist = nltk.FreqDist(w.lower() for w in tokens)
    allWordExceptStopDist = nltk.FreqDist(
        w.lower() for w in tokens if w not in stopwords and len(w) >= 3 and w[0] not in char_blacklist)
    all_words = [i for i in allWordExceptStopDist]
    mostCommon = allWordExceptStopDist.most_common(top)
    freq_words += [word for word, number in mostCommon]

features = np.zeros(pow(len(tokens_list), 2) * top).reshape(len(tokens_list), len(tokens_list) * top)

print('FEATURES CREATION')
for index, line in enumerate(tokens_list):
    for word in line:
        if word in freq_words:
            features[index][freq_words.index(word)] = 1

print(labels)
print(labels.shape, features.shape)
c, r = labels.shape
labels = labels.reshape(c,)
lr = LogisticRegression()
prediction = cross_val_predict(lr, features, labels, cv=cv_number)
print(prediction)
print("Precision_score: {}".format(metrics.precision_score(labels, prediction, average='micro')))
print("Recall_score: {}".format(metrics.recall_score(labels, prediction, average='micro')))
print("F1 score: {}".format(metrics.f1_score(labels, prediction, average='micro')))

# with open("filtered_data_1000", 'a') as o1:
#     for row in filter_rows:
#         counter = 0
#         for word in row:
#             counter += 1
#             if counter == len(row):
#                 o1.write(word)
#             else: o1.write(word+',')
#         o1.write('\n')
