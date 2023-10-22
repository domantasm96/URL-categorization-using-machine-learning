# URL categorization using machine learning

Internet can be used as one important source of information for machine learning algorithms. Web
pages store diverse information about multiple domains. One critical problem is how to categorize
this information. 

Websites classification is performed by using NLP techniques that helps to generate words frequencies for each category and by calculating categories weights it is possible to predict categories for Websites. 

Main dataset for this project could be found: [URL categorization dataset file](https://data.world/crowdflower/url-categorization)

## Url category predictions usage

For url predictions you could use already generated words frequency model that was created at 2020-12-26: 
`frequency_mode/word_frequency_2021.pickle`

Otherwise, you could create your own model words frequency model by executing `construct_features.py`.

For python versions management I would highly advise to use [pyenv tool](https://github.com/pyenv/pyenv).

1. **Execute `poetry shell`**. [Poetry installation guide](https://python-poetry.org/docs/)
```commandline
foo@bar:~$ poetry shell
```
2. **Start FastAPI local server**
```commandline
foo@bar:~$ uvicorn url_predictions.api_main:app --reload
```
3. **There are two ways to get website category predictions:**

* _**Use curls commands to predict url:**_
```commandline
curl -X 'POST' \
  'http://localhost:8000/predict/?url=bbc.com' \
  -H 'accept: application/json' \
  -d ''
```

* _**Use FastAPI UI:**_
  1. Go to `http://localhost:8000/docs`
  2. Expand `/predict/` **POST** endpoint page
  3. Write an url and press `execute`
  4. You should get a JSON response with the results


### Prediction results structure

Request url: `http://localhost:8000/predict/?url=bbc.com`
Response:
```commandline
{
  "main_category": "News_and_Media",
  "category_weight": 7123532,
  "sub_category": "Reference",
  "sub_weight": 7038726,
  "response": "All HTML content",
  "tokens":  [
      "bbc",
      "homepage",
      "homepageaccessibility",
      "linksskip",
      "contentaccessibility",
      .
      .
      .
      ]
}
```
## Documentation

**Website Classification Using Machine Learning Approaches.pdf** 


This project is my Bachelor thesis, so it also has documentation part written with LaTeX. Documentation with original LaTeX files is located in the **Documentation** folder.

Please note that some concepts of documentation does not exist anymore or there are new things since some changes were applied after documentation was written.
