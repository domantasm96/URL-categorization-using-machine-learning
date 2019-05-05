import pandas as pd
import numpy as np
import os
import ast
from collections import Counter
import pickle

date = '2019-05-05'
input_path = f'Datasets/Translated_tokens_{date}.csv'
output_path = f'Models/Models_LR_{date}.joblib'
words_path = f'Frequency_models/word_frequency_2019-05-05.picle'
if os.path.isfile(input_path) and os.path.isfile(words_path) and not os.path.isfile(output_path):

    df = pd.read_csv(input_path)
    pickle_in = open(words_path,"rb")
    words_frequency = pickle.load(pickle_in)
    # Models creation
    top = 2500
    from collections import Counter

    features = np.zeros(df.shape[0] * top).reshape(df.shape[0], top)
    labels = np.zeros(df.shape[0])
    counter = 0
    for i, row in df.iterrows():
        c = [word for word, word_count in Counter(ast.literal_eval(row['tokens_en'])).most_common(top)]
        labels[counter] = list(set(df['main_category'].values)).index(row['main_category'])
        for word in c:
            if word in words_frequency[row['main_category']]:
                features[counter][words_frequency[row['main_category']].index(word)] = 1
        counter += 1

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
    lr_predictions = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    print('LogisticRegression')
    print('Score: ', score)
    print('Top: ', top)
    print('Dataset length: ', df.shape[0])
    print()

    from sklearn.svm import LinearSVC
    clf = LinearSVC()
    clf.fit(X_train, y_train)
    clf_predictions = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    print('SVM')
    print('Score: ', score)
    print('Top: ', top)
    print('Dataset length: ', df.shape[0])

    # Save models
    from sklearn.externals import joblib
    joblib.dump(lr, output_path)
