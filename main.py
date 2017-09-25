import datetime
from urllib.request import urlopen
import csv
import nltk
from bs4 import BeautifulSoup
from langdetect import detect

# nltk.download('punk')
file = '/home/domantas/Desktop/Erasmus/MachineLearning/url_classification_1project/URL-categorization-DFE.csv'
url_counter = 1
reader = csv.reader(open(file), delimiter=',')
data = []
header = next(reader)
char_blacklist = list(chr(i) for i in range(32, 127) if i <= 64 or i >= 91 and i <= 96 or i >= 123)
stopwords = nltk.corpus.stopwords.words('english')
stopwords.extend(char_blacklist)
language_blacklist = ['ru', 'vi', 'zh-cn', 'ja', 'ko', 'fa', 'ar', 'et', 'tr', 'bn', 'el', 'sk', 'sv', 'pl']
english_vocab = set(w.lower() for w in nltk.corpus.words.words())
category_dictionary = {}
counter = 1
print(datetime.datetime.now().time())

for row in reader:
    data.append(row)

for row in data:
    if counter == 15000:
        break
    print('Nr: ', counter)
    counter += 1
    if row[5] != 'Not_working' and float(row[6]) > 0.5:
        try:
            url = 'http://' + row[-1]
            html = urlopen(url, timeout=10).read()
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.extract()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk.lower() for chunk in chunks if chunk)

            text_vocab = set(w.lower() for w in text if w.lower().isalpha())
            if detect(text) != 'en':
                continue
            tokens = nltk.word_tokenize(text)
            try:
                category_dictionary[str(row[5])] += tokens
            except:
                category_dictionary[str(row[5])] = tokens
            print('----')
            print('URL: ' + row[-1])
            print('CATEGORY: ' + row[5])
            print(url_counter)
            url_counter += 1

        except:
            pass
category_result = {}
for key in category_dictionary:
    print(key)
    tokens = category_dictionary[key]
    allWordDist = nltk.FreqDist(w.lower() for w in tokens)
    allWordExceptStopDist = nltk.FreqDist(
        w.lower() for w in tokens if w not in stopwords and len(w) >= 3 and w[0] not in char_blacklist)
    all_words = [i for i in allWordExceptStopDist]
    mostCommon = allWordExceptStopDist.most_common(50)
    category_result[key] = mostCommon
    print(mostCommon)
# print(category_result)
print(datetime.datetime.now().time())