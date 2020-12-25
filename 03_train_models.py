import pandas as pd
import numpy as np
import ast
import pickle
import joblib
from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d")
input_path = f'Datasets/Translated_tokens_{date}.csv'
output_path = f'Models/Models_LR_{date}.joblib'
words_path = f'Frequency_models/word_frequency_{date}.picle'

df = pd.read_csv(input_path)
pickle_in = open(words_path, "rb")
words_frequency = pickle.load(pickle_in)
# Models creation
top = 20000
from collections import Counter

features = np.zeros(df.shape[0] * top).reshape(df.shape[0], top)
labels = np.zeros(df.shape[0])
counter = 0

print('Generating features')
all_categories = list(df.main_category.unique())
for i, row in df.iterrows():
    c = [word for word, word_count in Counter(ast.literal_eval(row['tokens_en'])).most_common(top)]
    labels[counter] = all_categories.index(row['main_category'])
    for word in c:
        if word in words_frequency[row['main_category']]:
            features[counter][words_frequency[row['main_category']].index(word)] = 1
    counter += 1
print("Features generation done")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

print(len(X_train), len(X_test), len(y_train), len(y_test))
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

# Save models
joblib.dump(lr, output_path)
