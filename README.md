# Python-ML-Pipeline
Disaster Response Pipeline for Classifying Messages

## Project Components
1. ETL Pipeline
2. ML Pipeline
3. Flask Web App

## File Description
> *disaster_messages.csv*: contains raw message data<br>
> *disaster_categories.csv*: contains raw category data<br>
> ETL Pipeline Preparation.ipynb: ETL Implementation process analysis and explanation
> ML Pipeline Preparation.ipynb: ML Implementation process analysis and explanation
> process_data.py: Loads the messages and categories datasets, merges the two datasets, cleans the data, stores it in a SQLite database <br>
> train_classifier.py: Loads data from the SQLite database, splits the dataset into training and test sets, builds a text processing and machine learning pipeline,
trains and tunes a model using GridSearchCV, outputs results on the test set, exports the final model as a pickle file <br>

