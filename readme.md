# URL categorization using machine learning

Internet can be used as one important source of information for machine learning algorithms. Web
pages store diverse information about multiple domains. One critical problem is how to categorize
this information. Support vector machines and other supervised classification algorithms have been applied to text categorization.

## How to start a project

If you are running this project at a first time, first of all you need to process 3 steps:

1) **01_construct_features.py** - Scrape each URL and tokenize their HTML information

2) **02_construct_models.py** - Preprocess scrapped HTML data (Stopwords cleaning, translate non english words, excluding non english words). This python script also creates **most frequent words model for each category**.

3) **03_train_models.py** - Calculate most frequent words for each category and predict category. 
In order to run these python codes, you could do that manually or by executing bash script (**construct_data.sh**)

```console
foo@bar:~$ chmod +x construct_data.sh
foo@bar:~$ ./construct_data.sh
```

Make sure, that all libraries are installed from **requirements.txt** file:

```console
foo@bar:~$ pip install -r requirements.txt
```


## Predictions

In order to predict custom URL categories, **Words Frequency model** should be created and it is located in the **Frequency_models** directory. Usually it is saved in *pickle* format. 

**Words Frequency model for each category** is created by executing **02_construct_models.py** python script.


For custom URL predictions I would suggest to use Jupyter-Notebooks/Labs written codes, which are located in:

1) **Jupyter-notebook/predictions_test.ipynb** - Predicts custom URL

2) **Jupyter-notebook/Predictions.ipynb** - Predicts array of URL's


## Documentation

**Website Classification Using Machine Learning Approaches.pdf** 


This project is my Bachelor thesis, so it also has documentation part written with LaTeX. Documentation with original LaTeX files is located in the **Documentation** folder.


## Contact information

### Linkedin https://www.linkedin.com/in/domantas-meidus-49089a133/

If you have any questions or suggestions related with this project, please contact me directly via linkedin or raise an issue.


Have a good day/night/evening/weekend! 
