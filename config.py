from datetime import datetime
from nltk.corpus import stopwords

date = datetime.now().strftime("%Y-%m-%d")

# Request headers parameters
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive'}

# Paths for datasets and models
MAIN_DATASET_PATH = f'Datasets/url_categorization_dfe.csv'
TOKENS_PATH = f'Datasets/Feature_dataset_{date}.csv'
WORDS_FREQUENCY_PATH = f"Frequency_models/word_frequency_{date}.picle"


TOP_LEVEL_DOMAIN_WHITELIST = {'com', 'net', 'to', 'info', 'org', 'cn', 'jp', 'tw', 'ir', 'uk', 'ae', 'tv', 'in', 'hk',
                              'th', 'ca', 'us', 'gr', 'ws', 'io', 'bg', 'au', 'gov', 'il', 'za', 'edu', 'me', 'ph',
                              'ag', 'bd', 'biz', 'ie', 'kr', 'asia', 'my', 'by', 'nz', 'mil', 'sg', 'kz', 'eg', 'qa',
                              'pn', 'guide', 'ke', 'bz', 'im', 'pk', 'ps', 'aero', 'ck', 'museum', 'int', 'np', 'jobs',
                              'cy', 'gg', 'bw', 'bt', 'gh', 'af', 'coop', 'mk', 'tk'
                              }

STOPWORDS = set(stopwords.words('english'))
with open("Datasets/stopwords_extended.txt") as f:
    for word in f:
        STOPWORDS.add(word.replace('\n', ''))

for tld in (TOP_LEVEL_DOMAIN_WHITELIST - {'museum', 'jobs', 'guide', 'aero'}):
    STOPWORDS.add(tld)

# Parameter for words frequency model generation
FREQUENCY_TOP_WORDS = 20000

THREADING_WORKERS = 16
MULTIPROCESSING_WORKERS = 6
