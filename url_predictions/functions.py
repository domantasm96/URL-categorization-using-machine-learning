import asyncio
import pickle
import re
from typing import Any

import aiohttp
import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from url_predictions.config import FREQUENCY_TOP_WORDS, REQUEST_HEADERS, STOPWORDS, logger

wnl = WordNetLemmatizer()


def predict_category(
    words_frequency: dict[str, str], html_content: tuple[int | str | None, list[str]]
) -> dict[str, str | int]:
    tokens = html_content[1]
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
    return {
        "main_category": main_category,
        "category_weight": category_weight,
        "sub_category": main_category_2,
        "sub_weight": category_weight_2,
    }


def remove_stopwords(tokens: list[str]) -> list[str]:
    tokens_list = []
    for word in tokens:
        word = wnl.lemmatize(word.lower())
        if word not in STOPWORDS:
            tokens_list.append(word)
    return list(filter(lambda x: len(x) > 1, tokens_list))


def scrape_url(url: str, prediction: bool = False) -> str | None:
    try:
        return requests.get(url, headers=REQUEST_HEADERS, timeout=15).text
    except requests.exceptions.RequestException as e:
        logger.error(e)
        if prediction:
            raise e
        return None


async def fetch_url(url: str, session: Any) -> str | None:
    try:
        async with session.get(url) as response:
            return await response.text
    except aiohttp.ClientError as e:
        logger.error(e)
        return None


async def fetch_html_content_async(urls: list[str]) -> Any:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(url, session) for url in urls]
        html_contents = await asyncio.gather(*tasks)
        return html_contents


def fetch_html_content_sync(urls: list[str]) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    html_contents = loop.run_until_complete(fetch_html_content_async(urls))
    return html_contents


def parse_response(response: list[int | str | None]) -> tuple[int | str | None, list[str]]:
    index = response[0]
    html_content = response[1]
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        [tag.decompose() for tag in soup("script")]  # pylint: disable=expression-not-assigned
        [tag.decompose() for tag in soup("style")]  # pylint: disable=expression-not-assigned
        text = soup.get_text()
        cleaned_text = re.sub("[^a-zA-Z]+", " ", text).strip()
        tokens = word_tokenize(cleaned_text)
        tokens_lemmatize = remove_stopwords(tokens)
        return index, tokens_lemmatize
    return index, [""]


def save_to_pickle(target: Any, output_path: str, write_mode: str) -> None:
    with open(output_path, write_mode) as pickle_out:  # pylint: disable=unspecified-encoding
        pickle.dump(target, pickle_out)


def read_pickle(input_path: str) -> Any:
    with open(input_path, "rb") as pickle_in:
        return pickle.load(pickle_in)


def format_url(url: str) -> str:
    if not re.match("(?:http|ftp|https)://", url):
        return f"https://{url}"
    return url
