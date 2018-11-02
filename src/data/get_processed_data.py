import pandas as pd
import numpy as np
import os

def read_data():
    # set the path of the raw data
    raw_data_path = os.path.join(os.path.pardir, 'data', 'raw')
    train_file_path = os.path.join(raw_data_path, 'train.csv')
    test_file_path = os.path.join(raw_data_path, 'test.csv')
    # read the data with default parameters
    train_df = pd.read_csv(train_file_path, index_col = 'PassengerId')
    test_df = pd.read_csv(test_file_path, index_col = 'PassengerId')
    # combine train and test datasets
    test_df['Survived'] = -1
    df = pd.concat((train_df, test_df), axis = 0, sort = False)
    return df

def process_data(df):
    # using method chaining concept
    return (df
            # create Title feature
            .assign(Title = lambda x: x.Name.map(get_title))
            # fill missing values
            .pipe(fill_missing_values)
            # create Fare_Bin feature
            .assign(Fare_Bin = lambda x: pd.qcut(x.Fare, 4, labels = ['very_low', 'low', 'high', 'very_high']))
            # create AgeState feature
            .assign(AgeState = lambda x: np.where(x.Age >= 18, 'Adult', 'Child'))
            # create FamilySize feature
            .assign(FamilySize = lambda x: x.Parch + x.SibSp + 1)
            # create IsMother feature
            .assign(IsMother = lambda x: np.where(((x.Age > 18) & (x.Sex == 'female') & (x.Parch > 0) & (x.Title != 'Miss')), 1, 0))
            # create Deck feature
            .assign(Cabin = lambda x: np.where(x.Cabin == 'T', np.NaN, x.Cabin))
            .assign(Deck = lambda x: x.Cabin.map(get_deck))
            # feature encoding
            .assign(IsMale = lambda x: np.where(x.Sex == 'male', 1, 0))
            .assign(IsAdult = lambda x: np.where(x.AgeState == 'Adult', 1, 0))
            .pipe(pd.get_dummies, columns = ['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked'])
            # drop unnecessary columns
            .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex', 'AgeState'], axis = 1)
            # reorder columns
            .pipe(reorder_columns)
           )

def get_title(name):
    title_group = {
        'mr': 'Mr',
        'mrs': 'Mrs',
        'miss': 'Miss',
        'master': 'Master',
        'don': 'Sir',
        'rev': 'Sir',
        'dr': 'Officer',
        'mme': 'Mrs',
        'ms': 'Mrs',
        'major': 'Officer',
        'lady': 'Lady',
        'sir': 'Sir',
        'mlle': 'Miss',
        'col': 'Officer',
        'capt': 'Officer',
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'
    }
    first_name_with_title = name.split(',')[1]
    title = first_name_with_title.split('.')[0]
    title = title.strip().lower()
    return title_group[title]

def get_deck(cabin):
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def fill_missing_values(df):
    # embarked
    df.Embarked.fillna('C', inplace = True)
    # fare
    median_Fare = df[((df.Pclass == 3) & (df.Embarked == 'S'))].Fare.median()
    df.Fare.fillna(median_Fare, inplace = True)
    # age
    age_title_median = df.groupby(['Title']).Age.transform('median')
    df.Age.fillna(age_title_median, inplace = True)
    return df

def reorder_columns(df):
    columns = [column for column in df.columns if column != 'Survived']
    columns = ['Survived'] + columns
    df = df[columns]
    return df

def write_data(df):
    processed_data_path = os.path.join(os.path.pardir, 'data', 'processed')
    write_train_path = os.path.join(processed_data_path, 'train.csv')
    write_test_path = os.path.join(processed_data_path, 'test.csv')
    # train data
    df.loc[df.Survived != -1].to_csv(write_train_path)
    # test data
    columns = [column for column in df.columns if column != 'Survived']
    df.loc[df.Survived == -1, columns].to_csv(write_test_path)

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)