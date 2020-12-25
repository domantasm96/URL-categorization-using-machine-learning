# System
import ast
import math
import pickle
import os
from datetime import datetime
# ML
import numpy as np
import pandas as pd
# WEB
# NLTK
import nltk
from nltk.stem import WordNetLemmatizer
# Selenium
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By


def remove_stopwords(tokens):
    for i, token in enumerate(tokens):
        tokens[i] = ''.join([ch for ch in token if ch not in char_blacklist])
    tokens_sw = [w.lower() for w in tokens if w not in stopwords]
    tokens_lemmatize = [wnl.lemmatize(token) for token in tokens_sw]
    return list(filter(lambda x: len(x) > 2, tokens_lemmatize))


def get_en_words(tokens_lemmatize):
    english_tokens = []
    for word in tokens_lemmatize:
        english_tokens.append(word) if word in english_vocab else ''
    english_confidence = round(len(english_tokens) / len(tokens_lemmatize) * 100, 2) if len(english_tokens) > 0 else 0
    return english_tokens, english_confidence


def translate_words():
    foreign_words = list(set(tokens_lemmatize) - set(en_tokens))
    translated_words = []
    if len(foreign_words):
        chunk_size = math.ceil(len(foreign_words) / 5000)
        chunks = np.array_split(foreign_words, chunk_size)
        for chunk in chunks:
            foreign_text = " ".join(chunk)
            input_box = driver.find_element_by_id('source')
            driver.execute_script(f"document.getElementById('source').value = '{foreign_text}';")
            # input_box.send_keys(foreign_text)
            WebDriverWait(driver, 20).until(
                ec.visibility_of_element_located((By.CSS_SELECTOR, "span.tlid-translation.translation")))
            output_box = driver.find_element_by_css_selector("span.tlid-translation.translation").text
            translated_words.extend(output_box.split(' '))
            input_box.clear()
    return translated_words


start = datetime.now()
print(start)

date = datetime.now().strftime("%Y-%m-%d")
input_path = f'Datasets/Feature_dataset_{date}.csv'
output_path = f'Datasets/Translated_tokens_{date}.csv'
words_filename = f"Frequency_models/word_frequency_{date}.picle"

selenium_translation = False
selenium_driver_path = '/home/domantas/Documents/selenium_drivers/chromedriver'

print(os.path.isfile(input_path), os.path.isfile(output_path), os.path.isfile(words_filename))
nltk.download('stopwords')
nltk.download('words')
if selenium_translation:
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    driver = webdriver.Chrome(selenium_driver_path, chrome_options=options)
    driver.get("https://translate.google.com/")

english_vocab = set(w.lower() for w in nltk.corpus.words.words('en'))
english_tolerance = 50
english_confidence = []

char_blacklist = list(chr(i) for i in range(32, 127) if (i <= 47 or i >= 58) \
                      and (i <= 64 or i >= 91) and (i <= 96 or i >= 123))
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
top = 20000
hdr = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

df = pd.read_csv(input_path)
df = df[~df['tokens'].isnull()]
df['tokens'] = df['tokens'].map(lambda x: ast.literal_eval(x))
df['tokens_en'] = ''
df['en_confidence'] = ''
counter = 0
wnl = WordNetLemmatizer()
for row_id, row in df.iterrows():
    counter += 1
    try:
        tokens_lemmatize = remove_stopwords(row['tokens'])
        en_tokens, en_confidence = get_en_words(tokens_lemmatize)
        if selenium_translation:
            translated_words = translate_words()
            en_tokens_tr, en_confidence_tr = get_en_words(translated_words)
            en_tokens.extend(remove_stopwords(en_tokens_tr))
        df.at[row_id, 'tokens_en'] = en_tokens if len(en_tokens) else ''
        df.at[row_id, 'en_confidence'] = round(len(en_tokens) / len(tokens_lemmatize) * 100, 2) if len(
            en_tokens) > 0 else 0
        print(f"{counter}/{df.shape[0]} || {row['url']}")
    except Exception as e:
        print(f"{counter}/{df.shape[0]} || FAILED. {row['url']}", e)
        #         print(traceback.print_exc())
        continue

if selenium_translation:
    driver.close()

stop = datetime.now()
exec_time = stop - start

print(exec_time)
df = df[df['tokens_en'] != '']
df.to_csv(output_path, index=False)

words_frequency = {}
for category in df.main_category.unique():
    print(category)
    all_words = []
    df_temp = df[df.main_category == category]
    for word in df_temp.tokens_en:
        all_words.extend(word)
    most_common = [word[0] for word in nltk.FreqDist(all_words).most_common(top)]
    words_frequency[category] = most_common

# Save words_frequency model
pickle_out = open(words_filename, "wb")
pickle.dump(words_frequency, pickle_out)
pickle_out.close()
