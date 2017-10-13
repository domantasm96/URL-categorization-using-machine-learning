import datetime
import csv
import nltk
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
from langdetect import detect
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import pandas as pd


def score_calculation(labels, prediction):
    y_true = pd.Series(labels)
    y_pred = pd.Series(prediction)
    print(str(lr))
    print('Confusion matrix: \n{}'.format(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)))
    print(classification_report(labels, prediction))
    print("Accuracy score: {}".format(metrics.accuracy_score(labels, prediction)))

file = 'URL-categorization-DFE.csv'
limiter = 5000
cv_number = 5
top = 15
reader = csv.reader(open(file), delimiter=',')
header = next(reader)
char_blacklist = list(chr(i) for i in range(32, 127) if i <= 64 or i >= 91 and i <= 96 or i >= 123)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
language_whitelist = ['en']
domains_whitelist = ['com', 'org', 'net', '.us', '.uk', '.au', '.ca']
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
print(datetime.datetime.now().time())
data = []
for row in reader:
    data.append(row)

tokens_list = []
filter_data = []
counter = 0
print('URL parsing and filtering')
for url_counter, row in enumerate(data):
    if url_counter > limiter:
        break
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
            if detect(text) not in language_whitelist or (row[-1][-3:] not in domains_whitelist and row[-1][-3:] not in domains_whitelist):
                continue
            counter += 1
            tokens = nltk.word_tokenize(text)
            tokens_list += [nltk.word_tokenize(text)]
            print('{} | {} |URL: {}| CATEGORY: {}'.format(url_counter, counter, row[-1], row[5]))
            filter_data += [row[5]]

        except:
            pass
print('Filtered English URL')
f1 = nltk.FreqDist(filter_data).most_common()
f2 = list(category for category, number in f1 if number >= cv_number)
all_categories = list(set(f2))
labels_data = [index for index, word in enumerate(f2)]
print(labels_data)
print('CREATING LABELS DATA.')
labels = []
counter = 0
for index, word in enumerate(filter_data):
    if word in all_categories:
        labels += [all_categories.index(word)]
    else:
        tokens_list.pop(index - counter)
        counter += 1
save = labels
labels = np.array(labels).reshape((len(labels), 1))
print('CREATING FREQUENT WORDS LIST..')
freq_words = []
for tokens in tokens_list:
    allWordDist = nltk.FreqDist(w.lower() for w in tokens)
    allWordExceptStopDist = nltk.FreqDist(
        w.lower() for w in tokens if w not in stopwords and len(w) >= 3 and w[0] not in char_blacklist)
    all_words = [i for i in allWordExceptStopDist]
    mostCommon = allWordExceptStopDist.most_common(top)
    freq_words += [word for word, number in mostCommon]

print('CREATING FEATURES DATA...')
features = np.zeros(pow(len(tokens_list), 2) * top).reshape(len(tokens_list), len(tokens_list) * top)
for index, line in enumerate(tokens_list):
    for word in line:
        if word in freq_words:
            features[index][freq_words.index(word)] = 1

for number, word in enumerate(all_categories):
    print(number, word)
c, r = labels.shape
print('Number of URL: {}'.format(c))
labels = labels.reshape(c,)
print('************ Logistic Regression ************')
lr = LogisticRegression()
prediction = cross_val_predict(lr, features, labels, cv=cv_number)
score_calculation(labels, prediction)
print('************ Decision Tree ************')
lr = DecisionTreeClassifier()
prediction = cross_val_predict(lr, features, labels, cv=cv_number)
score_calculation(labels, prediction)
print('************ KNeighbors ************')
lr = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
prediction = cross_val_predict(lr, features, labels, cv=cv_number)
score_calculation(labels, prediction)

# print('************ Support Vector Machine ************')
# lr = SVC()
# prediction = cross_val_predict(lr, features, labels, cv=cv_number)
# score_calculation(labels, prediction)
print(datetime.datetime.now().time())
