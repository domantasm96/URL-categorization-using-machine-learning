import pandas as pd
import nltk
import os
from datetime import datetime
import requests
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


def remove_stopwords(tokens):
    for i, token in enumerate(tokens):
        tokens[i] = ''.join([ch for ch in token if ch not in char_blacklist])
    tokens_sw = [w.lower() for w in tokens if w not in stopwords]
    tokens_lemmatize = [wnl.lemmatize(token) for token in tokens_sw]
    return tokens_lemmatize


def scrape(url):
    print(url[0], url[1])
    try:
        return requests.get(url[1], headers=request_headers, timeout=15)
    except:
        return ''


date = datetime.now().strftime("%Y-%m-%d")
input_path = f'Datasets/url_categorization_dfe.csv'
output_path = f'Datasets/Feature_dataset_{date}.csv'

request_headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

wnl = WordNetLemmatizer()
char_blacklist = list(
    chr(i) for i in range(32, 127) if (i <= 47 or i >= 58) and (i <= 64 or i >= 91) and (i <= 96 or i >= 123))
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)

if not os.path.isfile(output_path):
    df = pd.read_csv(input_path)[['url', 'main_category', 'main_category_confidence']]
    df = df[(df['main_category'] != 'Not_working') & (df['main_category_confidence'] >= 0.5)].reset_index(drop=True)
    df['url'] = df['url'].apply(lambda x: 'http://' + x)
    df['tokens'] = ''
    with ThreadPoolExecutor(16) as executor:
        start = datetime.now()
        print("Scraping begins: ", start)
        results = executor.map(scrape, [(i, elem) for i, elem in enumerate(df['url'].values)])
    stop = datetime.now()
    exec_time = stop - start
    print('Scraping finished. Time: ', exec_time)

    for i, res in enumerate(results):
        if res != '' and res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            [tag.decompose() for tag in soup("script")]
            [tag.decompose() for tag in soup("style")]
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk.lower() for chunk in chunks if chunk)
            # Tokenize text
            tokens = [token.lower() for token in word_tokenize(text)]
            # Remove stopwords
            tokens_lemmatize = remove_stopwords(tokens)
            df.at[i, 'tokens'] = tokens_lemmatize
        else:
            df.at[i, 'tokens'] = ['']

    df.to_csv(output_path, index=False)
