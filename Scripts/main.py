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
nltk.download('wordnet')

month = "january"
char_blacklist = list(chr(i) for i in range(32, 127) if i <= 64 or i >= 91 and i <= 96 or i >= 123)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
english_tolerance = 50
english_confidence = []
words_threshold = 10
top = 2500
toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
words_frequency = {}

def no_filter_data():
    file = '../Datasets/URL-categorization-DFE.csv'
    df = pd.read_csv(file)[['main_category', 'main_category:confidence', 'url']]
    df = df[(df['main_category'] != 'Not_working') & (df['main_category:confidence'] > 0.5)]
    df['tokenized_words'] = ''
    counter = 0
    for i, row in df.iterrows():
        counter += 1
        print("{}, {}/{}; Time: {}".format(row['url'], counter, len(df), str(datetime.now())))
        try:
            hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
               'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
               'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
               'Accept-Encoding': 'none',
               'Accept-Language': 'en-US,en;q=0.8',
               'Connection': 'keep-alive'}
            req = urllib.request.Request('http://' + row['url'], headers=hdr)
            html = urlopen(req, timeout=15).read()
        except:
            print("{} Failed".format(row['url']))
            continue

        soup = BeautifulSoup(html, "html.parser")
        [tag.decompose() for tag in soup("script")]
        [tag.decompose() for tag in soup("style")]
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk.lower() for chunk in chunks if chunk)
        # Tokenize text
        tokens = [token.lower() for token in toker.tokenize(text)]
        # Remove stopwords
        wnl = WordNetLemmatizer()
        tokens = [token.lower() for token in toker.tokenize(text)]
        tokens_stopwords = [w.lower() for w in tokens if w not in stopwords and len(w) >= 3 and w[0] not in char_blacklist]
        tokens_lemmatize = [wnl.lemmatize(token) for token in tokens_stopwords]
        # Calculate percentage of english words
        english_tokens = []
        for word in tokens_lemmatize:
            english_tokens.append(word.lower()) if word.lower() in english_vocab else ''
        english_confidence = len(english_tokens) / len(tokens_lemmatize) * 100 if len(english_tokens) > 0 else 0
        if len(english_tokens) < words_threshold or english_confidence < english_tolerance:
            continue

        df.at[i, 'tokenized_words'] = english_tokens if english_confidence > english_tolerance else ''
        df.at[i, 'english:confidence'] = english_confidence

    df = df[df['tokenized_words'] != '']
    df.to_csv("../Datasets/full_data_{}.csv".format(month))

    # Read new generated data set file
    df = pd.read_csv("../Datasets/full_data_{}.csv".format(month))
    # Generate most frequent words list for each category
    words_frequency = {}
    for category in set(df['main_category'].values):
        print(category)
        all_words = []
        for row in df[df['main_category'] == category]['tokenized_words'].tolist():
            for word in ast.literal_eval(row):
                all_words.append(word)
        most_common = nltk.FreqDist(w for w in all_words).most_common(top)
        words_frequency[category] = most_common
    # Extract only words
    for category in set(df['main_category'].values):
        words_frequency[category] = [word for word, number in words_frequency[category]]

    # Create labels and features set for ML
    features = np.zeros(df.shape[0] * top).reshape(df.shape[0], top)
    labels = np.zeros(df.shape[0])
    counter = 0
    for i, row in df.iterrows():
        c = [word for word, word_count in Counter(ast.literal_eval(row['tokenized_words'])).most_common(top)]
        labels[counter] = list(set(df['main_category'].values)).index(row['main_category'])
        for word in c:
            if word in words_frequency[row['main_category']]:
                features[counter][words_frequency[row['main_category']].index(word)] = 1
        counter += 1
    # Features and labels splitting to training and testing data
    from sklearn.metrics import accuracy_score
    from scipy.sparse import coo_matrix
    X_sparse = coo_matrix(features)

    from sklearn.utils import shuffle
    X, X_sparse, y = shuffle(features, X_sparse, labels, random_state=0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    print('LogisticRegression')
    print('Score: ', score)
    print('Top: ', top)
    print('Dataset length: ', df.shape[0])
    print()

    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print('SVM')
    print('Score: ', score)
    print('Top: ', top)
    print('Dataset length: ', df.shape[0])

    # Save models
    from sklearn.externals import joblib
    filename = "../Models/{}/LR_model_{}.joblib".format(month.title(), month)
    if not os.path.isfile(filename):
        joblib.dump(lr, filename)

    filename = "../Models/{}/LSVM_model_{}.joblib".format(month.title(), month)
    if not os.path.isfile(filename):
        joblib.dump(clf, filename)

    # Save words_frequency model
    import pickle
    words_filename = "../Models/{}/word_frequency_{}.picle".format(month.title(), month)
    if not os.path.isfile(words_filename):
        pickle_out = open(words_filename,"wb")
        pickle.dump(words_frequency, pickle_out)
        pickle_out.close()


if not os.path.isfile("../Datasets/full_data_{}.csv".format(month.title(), month)):
    no_filter_data()
