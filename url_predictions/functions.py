import pickle
import re
from typing import Any

import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from requests import Response

from url_predictions.config import FREQUENCY_TOP_WORDS, REQUEST_HEADERS, STOPWORDS, logger

wnl = WordNetLemmatizer()


def predict_category(words_frequency: dict[str, str], tokens: list[str]) -> Any:
    category_weights = []
    for category in words_frequency:
        weight = 0
        intersect_words = set(words_frequency[category]).intersection(set(tokens))
        for word in intersect_words:
            if word in tokens:
                index = words_frequency[category].index(word)
                weight += FREQUENCY_TOP_WORDS - index
        category_weights.append(weight)

    category_index = category_weights.index(max(category_weights))
    main_category = list(words_frequency.keys())[category_index]
    category_weight = max(category_weights)
    category_weights[category_index] = 0
    category_index = category_weights.index(max(category_weights))
    main_category_2 = list(words_frequency.keys())[category_index]
    category_weight_2 = max(category_weights)
    return main_category, category_weight, main_category_2, category_weight_2


def remove_stopwords(tokens: list[str]) -> list[str]:
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in STOPWORDS:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))


def scrape(url: str) -> Response | str:
    try:
        return requests.get(url, headers=REQUEST_HEADERS, timeout=15)
    except requests.exceptions.RequestException as e:
        logger.error(e)
        return ""


def parse_request(res: Response) -> list[str]:
    if res != "" and res.status_code == 200:
        soup = BeautifulSoup(res.text, "html.parser")
        [tag.decompose() for tag in soup("script")]  # pylint: disable=expression-not-assigned
        [tag.decompose() for tag in soup("style")]  # pylint: disable=expression-not-assigned
        text = soup.get_text()
        cleaned_text = re.sub("[^a-zA-Z]+", " ", text).strip()
        tokens = word_tokenize(cleaned_text)
        tokens_lemmatize = remove_stopwords(tokens)
        return tokens_lemmatize
    return [""]


def save_to_pickle(target: Any, output_path: str, write_mode: str) -> None:
    pickle_out = open(output_path, write_mode)  # pylint: disable=unspecified-encoding, consider-using-with
    pickle.dump(target, pickle_out)
    pickle_out.close()
