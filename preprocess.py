import math
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
    df.to_csv(output_file, index=False, sep=';')

def concat_subject_and_body(df, column_name):
    df[column_name] = df['MailSubject'] + ' ' + df['MailTextBody']
    return df

def split_long_rows(df, column_name, longer_than):
    df['length'] = df[column_name].str.len()

    copies = []
    for i in range(len(df)):
        text = str(df.loc[i, column_name])
        length = float(df.loc[i, 'length'])
        if length > longer_than:
            part_count = math.ceil(length / longer_than)

            for a in range(part_count):
                part_of_text = text[a * longer_than:(a*longer_than) + longer_than]
                copy = df.loc[i].copy()
                copy[column_name] = part_of_text
                copies.append(copy)

    temp_df = pd.DataFrame(copies)

    # Drop rows that are longer than longer_than
    df = df[df['length'] <= longer_than]

    # Join DFs
    df = df.append(temp_df)
    df = df.drop(columns=['length'])
    return df

def preprocess_and_produce_dfs(input_file='data/train.csv'):
    df = load_data(input_file)

    # 1: Simple concatenation (subject + body)
    df = concat_subject_and_body(df, 'input_1')

    # 2: Split every row that has more characters than 75th percentile
    df = concat_subject_and_body(df, 'input_2')
    max_chars = int(df['input_2'].str.len().quantile(q=0.75))
    df = split_long_rows(df, 'input_2', longer_than=max_chars)

    # Split and save
    train, test = train_test_split(df, test_size=0.2, random_state=328)
    save_df(train, 'train_trans.csv')
    save_df(test, 'test_trans.csv')

if __name__ == '__main__':
    preprocess_and_produce_dfs()
