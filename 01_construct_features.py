import pandas as pd
import urllib.request
import nltk
import os
from datetime import datetime
from urllib.request import urlopen
from multiprocessing import Pool, cpu_count
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup

def remove_stopwords(tokens):
    for i, token in enumerate(tokens):
        tokens[i] = ''.join([ch for ch in token if ch not in char_blacklist])
    tokens_sw = [w.lower() for w in tokens if w not in stopwords]
    tokens_lemmatize = [wnl.lemmatize(token) for token in tokens_sw]
    return tokens_lemmatize


def crawl(object):
    print(f"{object['i']}/{len(urls)} || {object['url']}")
    tokens_lemmatize = ''
    try:
        req = urllib.request.Request(object['url'], headers=hdr)
        html = urlopen(req, timeout=15).read()
        soup = BeautifulSoup(html, "html.parser")
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
    except Exception:
        print(f"{object['i']}/{len(urls)} || {object['url']} FAILED. ")
    return tokens_lemmatize if len(tokens_lemmatize) else ''
    #     return page_tokens[object['i']]

date = datetime.now().strftime("%Y-%M-%d")
input_path = f'Datasets/URL-categorization-DFE.csv'
output_path = f'Datasets/Feature_dataset_{date}.csv'
if not os.path.isfile(output_path):
    df = pd.read_csv(input_path)[['url', 'main_category', 'main_category:confidence']]
    df = df[(df['main_category'] != 'Not_working') & (df['main_category:confidence'] >= 0.5)]
    df['url'] = df['url'].map(lambda x: 'http://' + x)

    hdr = {
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

    start = datetime.now()
    print(start)
    urls = [{"i": index, "url": url} for index, url in enumerate(list(df['url'].values))]
    df['tokens'] = ''
    p = Pool(cpu_count() * 2)
    tokens = p.map(crawl, urls)
    df['tokens'][:len(tokens)] = tokens
    stop = datetime.now()
    print(stop)
    exec_time = stop - start
    print(exec_time)
    df.to_csv(output_path, index=False)
