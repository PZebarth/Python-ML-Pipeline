# Python-ML-Pipeline
Disaster Response Pipeline for Classifying Messages

## Project Description
This project will analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. <br>

You'll find a data set containing real messages that were sent during disaster events. I have created a machine learning pipeline to categorize these events so that  messages can be sent to an appropriate disaster relief agency. <br>

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. <br>

## Project Components
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

## File Description
> * **disaster_messages.csv**: contains raw message data<br>
> * **disaster_categories.csv**: contains raw category data<br>
> * **ETL Pipeline Preparation.ipynb**: ETL Implementation process analysis and explanation
> * **ML Pipeline Preparation.ipynb**: ML Implementation process analysis and explanation
> * **process_data.py**: Loads the messages and categories datasets, merges the two datasets, cleans the data, stores it in a SQLite database <br>
> * **train_classifier.py**: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file <br>

## Using the files

* 1. To run ETL pipeline: data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response_db.db
* 2. To run the ML pipeline: python models/train_classifier.py data/disaster_response_db.db models/classifier.pkl
* 3. To run web app: python app/run.py
