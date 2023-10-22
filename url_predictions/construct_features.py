from url_predictions.config import TOKENS_PATH, WORDS_FREQUENCY_PATH
from url_predictions.features import FeaturesClass
from url_predictions.functions import save_to_pickle


def main() -> None:
    features = FeaturesClass()
    features.preprocess_main_dataset()

    features.scrape_urls_async_mode()
    features.analyze_responses_normal_mode()

    features.df.to_csv(TOKENS_PATH, index=False)
    features.generate_words_frequency()
    save_to_pickle(features.words_frequency, WORDS_FREQUENCY_PATH, "wb")


if __name__ == "__main__":
    main()
