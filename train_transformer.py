import pandas as pd
from sklearn.model_selection import train_test_split

val_size = 0.3


def train_transformer_pipeline(data_dir):
    train_set = pd.read_csv(data_dir / 'train.csv', delimiter=';')
    labels = train_set[['ServiceProcessed', 'ManualGroups']].copy()
    train_set.drop(columns=['Impact', 'Urgency', 'IncidentType', 'ServiceProcessed', 'ManualGroups'], inplace=True)
    train_set['ServiceProcessed'] = labels['ServiceProcessed']
    train_set['ManualGroups'] = labels['ManualGroups']
    train_set.rename(columns={'Unnamed: 0': ''}, inplace=True)

    train_csv, validation_trans = train_test_split(train_set, test_size=val_size)
    train_csv.to_csv(data_dir / 'train_trans.csv', sep=';', index=False)
    validation_trans.to_csv(data_dir / 'validation_trans.csv', sep=';', index=False)

    test_set = pd.read_csv(data_dir / 'test_reduced.csv', delimiter=';')
    test_set['ServiceProcessed'] = ' '
    test_set['ManualGroups'] = ' '
    test_set.rename(columns={'Unnamed: 0': ''}, inplace=True)
    test_set.to_csv(data_dir / 'test_trans.csv', index=False, sep=';')
