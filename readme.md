# URL categorization using machine learning

Internet can be used as one important source of information for machine learning algorithms. Web
pages store diverse information about multiple domains. One critical problem is how to categorize
this information. 

Websites classification is performed by using NLP techniques that heps to generate words frequencies for each category and by calculating categories weights it is possible to predict categories for Websites. 

Main dataset for this project could be found: [URL categorization dataset file](https://data.world/crowdflower/url-categorization)

## How to start a project

If you are running this project at a first time, all you need is:
- download requirements for nltk module
- run **01_construct_features.py** file in order to generate **words_frequency** model for each category.

```console
foo@bar:~$ python -m nltk.downloader all
foo@bar:~$ python 01_construct_features.py
```
OR
```console
foo@bar:~$ chmod +x construct_data.sh
foo@bar:~$ ./construct_data.sh
```

**01_construct_features.py** execution time should be around *~45 mins*:

| Operation      | Execution time |
| ----------- | ----------- |
| Fetching responses for ~15k URLs      | ~40 min       |
| Analyzing responses and extracting tokens   | ~4 min        |
| Generating words_frequency for each category   | ~10 sec        |

### Config file
**config.py** file contains meta information for **parameters values, dataset path locations, output saving locations, stopwords list, threading/multiprocessing workers number parameters, frequency words parameter(this parameter impact category classification results)**. 

Make sure that **MAIN_DATASET_PATH** file path is correct before running **01_construct_features.py** file.

### Requirements
Make sure, that all libraries are installed from requirements.txt file:
```console
foo@bar:~$ pip install -r requirements.txt
```


## Predictions

Website category predictions could be done by executing **predict_url.py** file. There are two type arguments for **predict_url.py** file:
| Argument      | Description |
| ----------- | ----------- |
| -u (--url)      | Predict custom URL category       |
| -t (--text_file_path)   | Predict URLs from text file        |

**predict_url.py** file execution examples:
```console
Custom URL
foo@bar:~$ python predict_url.py -u https://www.bbc.com/news
```
```console
URLs prediction from text file
foo@bar:~$ python predict_url.py -t urls_to_predict.txt
```

## Documentation

**Website Classification Using Machine Learning Approaches.pdf** 


This project is my Bachelor thesis, so it also has documentation part written with LaTeX. Documentation with original LaTeX files is located in the **Documentation** folder.

Please note that some concepts of documentation do not exist anymore or there are new things since some changes were applied after documentation was written.

## Contact information

### Linkedin https://www.linkedin.com/in/domantas-meidus-49089a133/

If you have any questions or suggestions related with this project, please contact me directly via linkedin or raise an issue.


Have a good day/night/evening/weekend! 
