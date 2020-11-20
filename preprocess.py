import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path, sep=';')
    df.rename(columns={'Unnamed: 0': 'Unnamed'}, inplace=True)
    return df

def save_df(df, output_file):
    if output_file == None:
        print('Please provide an output file name')
        return
    df.to_csv(output_file, index=False)

def preprocess_and_produce_dfs(input_file='data/train.csv'):
    df = load_data(input_file)

    # 1: Simple concatenation (subject + body)
    df['input_1'] = df['MailSubject'] + ' ' + df['MailTextBody']

    # Split and save
    train, test = train_test_split(df, test_size=0.2, random_state=328)
    save_df(train, 'train_trans.csv')
    save_df(test, 'test_trans.csv')

if __name__ == '__main__':
    preprocess_and_produce_dfs()
