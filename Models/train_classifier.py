import sys
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle


def load_data(database_filepath):
    '''
    Loads sql database from filepath.

    Inputs:
    database_filepath - filepath string

    Outputs:
    X - input variable from database
    Y - outout variables from database
    category_names - category names of output variables from database

    '''
    # creating engine
    engine = create_engine(f'sqlite:///{database_filepath}')
    # creating dataframe from sql database
    df = pd.read_sql_table('messages_disaster', con=engine)
    # input variable
    X = df.message
    # outout variables
    Y = df.drop(['id','message','original','genre'], axis=1)
    # category names of output variables
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Takes text input and returns tokenized and lemmatized list of words in
    lower case with white space stripped.

    Input:
    text - string

    Output:
    clean_tokens - list of strings
    '''
    # tokenize text
    tokens = word_tokenize(text)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
    # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X,Y):
    '''
    Creates model from cross validation of parameters for pipeline.

    Output:
    cv - cross validation model
    '''
    # instantiating pipeline
    pipeline = Pipeline([
                        #('features', FeatureUnion([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        #]))
                        ])
    # creating parameters for cross validation
    parameters = {'clf__estimator__max_depth': [2, 4],
                  'clf__estimator__n_estimators': [5, 10],
                  'clf__estimator__min_samples_split': [2, 3]}
    # performing cross validation on pipeline
    cv = GridSearchCV(pipeline, param_grid = parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evalutes model with
    '''
    # creating predictions on test data input variables
    Y_pred = model.predict(X_test)

    # classification report for each category
    print(classification_report(Y_test, Y_pred, target_names = category_names))

def save_model(model, model_filepath):
    '''
    Saves model in pickle format.

    Input:
    model - supervised machine learning model
    model_filepath - filepath string
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
