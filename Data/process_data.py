import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads csv data from messages_filepath and categories_filepath filepaths and
    merges into a single dataframe.

    Input:
    messages_filepath - filepath string
    categories_filepath - filepath string

    Output:
    df - merged dataframe


    '''
    # reading csv data into dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merging data frames on id key
    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    '''
    Inserts correct column names, cleans data of duplicate and Nan values

    Input:
    df - dataframe

    Output:
    df - dataframe
    '''
    # splitting data in categories column
    categories = df.categories.str.split(pat =';', expand=True)
    # obtainging list of categories
    row = categories.iloc[0]
    # obatining clean list of categories
    category_colnames = row.apply(lambda x: x[:-2]).tolist()
    # renaming columns based on list of category names
    categories.columns = category_colnames
    # converting categories to numbers 0 and 1
    for column in categories:
        # selects last digit
        categories[column] = categories[column].apply(lambda x: x[-1:])
        # encodes data as integer
        categories[column] = categories[column].astype(int)

    # removes old categories column
    df = df.drop(columns = 'categories')
    # concatenates cleaned categories
    df = pd.concat([df,categories], axis=1)
    # drops duplicates
    df.drop_duplicates(inplace=True)
    # drops Nan values
    df.dropna(how = 'any', subset = category_colnames, inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves dataframe in sql table

    Input:
    df - dataframe
    database_filename - filename string
    '''
    #
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
