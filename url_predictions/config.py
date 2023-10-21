import logging

from nltk.corpus import stopwords

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG  # Set the logging level to DEBUG
)

# Create a logger for your module or script
logger = logging.getLogger(__name__)

# Request headers parameters
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",  # pylint: disable=line-too-long
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}

# Paths for datasets and models
MAIN_DATASET_PATH = "datasets/url_categorization_dfe.csv"
TOKENS_PATH = "datasets/feature_dataset.csv"
WORDS_FREQUENCY_PATH = "frequency_models/word_frequency.pickle"


TOP_LEVEL_DOMAIN_WHITELIST = {
    "com",
    "net",
    "to",
    "info",
    "org",
    "cn",
    "jp",
    "tw",
    "ir",
    "uk",
    "ae",
    "tv",
    "in",
    "hk",
    "th",
    "ca",
    "us",
    "gr",
    "ws",
    "io",
    "bg",
    "au",
    "gov",
    "il",
    "za",
    "edu",
    "me",
    "ph",
    "ag",
    "bd",
    "biz",
    "ie",
    "kr",
    "asia",
    "my",
    "by",
    "nz",
    "mil",
    "sg",
    "kz",
    "eg",
    "qa",
    "pn",
    "guide",
    "ke",
    "bz",
    "im",
    "pk",
    "ps",
    "aero",
    "ck",
    "museum",
    "int",
    "np",
    "jobs",
    "cy",
    "gg",
    "bw",
    "bt",
    "gh",
    "af",
    "coop",
    "mk",
    "tk",
}

STOPWORDS = set(stopwords.words("english"))
with open("datasets/stopwords_extended.txt", encoding="utf8") as f:
    for word in f:
        STOPWORDS.add(word.replace("\n", ""))

for tld in TOP_LEVEL_DOMAIN_WHITELIST - {"museum", "jobs", "guide", "aero"}:
    STOPWORDS.add(tld)

# Parameter for words frequency model generation
FREQUENCY_TOP_WORDS = 20000

THREADING_WORKERS = 10
MULTIPROCESSING_WORKERS = 6
