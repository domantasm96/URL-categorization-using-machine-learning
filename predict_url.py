import pickle
import config
import argparse
from functions import scrape_url

pickle_in = open(config.WORDS_FREQUENCY_PATH, "rb")
words_frequency = pickle.load(pickle_in)

parser = argparse.ArgumentParser(description='URLs for category predictions')
parser.add_argument('-u', '--url', help='Predict custom website')
parser.add_argument('-t', '--text_file_path', help='Predict websites written in text file')

args = parser.parse_args()

if args.url:
    url = args.url
    print(url)
    results = scrape_url(url, words_frequency)
    if results:
        print('Predicted main category:', results[0])
        print('Predicted submain category:', results[2])
elif args.text_file_path:
    file_path = args.text_file_path
    with open(file_path) as f:
        for url in f:
            url = url.replace('\n', '')
            print(url)
            results = scrape_url(url.replace('\n', ''), words_frequency)
            if results:
                print('Predicted main category:', results[0])
                print('Predicted submain category:', results[2])
else:
    parser.error("Please specify websites input type. More about input types you can find 'python predict_url -h'")
