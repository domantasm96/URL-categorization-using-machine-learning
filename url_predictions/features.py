from concurrent.futures import ThreadPoolExecutor
from typing import Any

import nltk
import pandas as pd

from url_predictions.config import (
    FREQUENCY_TOP_WORDS,
    MAIN_DATASET_PATH,
    MULTIPROCESSING_WORKERS,
    TOP_LEVEL_DOMAIN_WHITELIST,
)
from url_predictions.functions import fetch_html_content_sync, format_url, parse_response, scrape_url


class FeaturesClass:
    def __init__(self) -> None:
        self.df = pd.read_csv(MAIN_DATASET_PATH)
        self.url_responses: Any = []
        self.html_content: Any = []
        self.words_frequency: dict[str, list[str]] = {}

    def preprocess_main_dataset(self) -> None:
        self.df = self.df.rename(columns={"main_category:confidence": "main_category_confidence"})
        self.df = self.df[["url", "main_category", "main_category_confidence"]]
        self.df = self.df[(self.df["main_category"] != "Not_working") & (self.df["main_category_confidence"] >= 0.5)]
        self.df["url"] = self.df["url"].apply(format_url)
        self.df["tld"] = self.df.url.apply(lambda x: x.split(".")[-1])
        self.df = self.df[self.df.tld.isin(TOP_LEVEL_DOMAIN_WHITELIST)].reset_index(drop=True)
        self.df["tokens"] = ""

    def scrape_urls_normal_mode(self) -> None:
        self.url_responses = [(ind, scrape_url(url)) for ind, url in enumerate(self.df["url"].to_list())]

    def analyze_responses_normal_mode(self) -> None:
        self.html_content = [parse_response([ind, response]) for ind, response in enumerate(self.url_responses)]

        for ind, tokens in self.html_content:
            self.df.at[ind, "tokens"] = tokens

    def scrape_urls_async_mode(self) -> None:
        urls = self.df["url"].to_list()
        self.url_responses = fetch_html_content_sync(urls)

    def analyze_responses_multiprocessing_mode(self) -> None:
        with ThreadPoolExecutor(MULTIPROCESSING_WORKERS) as ex:
            self.html_content = ex.map(
                parse_response,
                [(i, elem) for i, elem in enumerate(self.url_responses)],  # pylint: disable=unnecessary-comprehension
            )

        for i, tokens in self.html_content:
            self.df.at[i, "tokens"] = tokens

    def generate_words_frequency(self) -> None:
        for category in self.df.main_category.unique():
            all_words = []
            df_temp = self.df[self.df.main_category == category]
            for word in df_temp.tokens:
                all_words.extend(word)
            most_common = [word[0] for word in nltk.FreqDist(all_words).most_common(FREQUENCY_TOP_WORDS)]
            self.words_frequency[category] = most_common
