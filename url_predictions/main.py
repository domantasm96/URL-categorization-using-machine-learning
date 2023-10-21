from url_predictions.config import TOKENS_PATH, WORDS_FREQUENCY_PATH
from url_predictions.features import FeaturesExtraction
from url_predictions.functions import save_to_pickle

features = FeaturesExtraction()
features.preprocess_main_dataset()
features.df = features.df[:100]
features.scrape_urls_normal_mode()
features.analyze_responses_normal_mode()
features.df.to_csv(TOKENS_PATH, index=False)
features.generate_words_frequency()
save_to_pickle(features.words_frequency, WORDS_FREQUENCY_PATH, "wb")
