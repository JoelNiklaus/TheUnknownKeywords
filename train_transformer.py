import pandas as pd

from bert import DATASET_DIR


def train_transformer_pipeline():
    train_set = pd.read_csv(DATASET_DIR / 'train.csv', delimiter=';')
    labels = train_set[['ServiceProcessed', 'ManualGroups']].copy()
    train_set.drop(columns=['Impact', 'Urgency', 'IncidentType', 'ServiceProcessed', 'ManualGroups'], inplace=True)
    train_set['ServiceProcessed'] = labels['ServiceProcessed']
    train_set['ManualGroups'] = labels['ManualGroups']
    train_set.rename(columns={'Unnamed: 0': ''}, inplace=True)
    train_set.to_csv(DATASET_DIR / 'train_trans.csv', sep=';', index=False)

    test_set = pd.read_csv(DATASET_DIR / 'test_reduced.csv', delimiter=';')
    test_set['ServiceProcessed'] = ' '
    test_set['ManualGroups'] = ' '
    test_set.rename(columns={'Unnamed: 0': ''}, inplace=True)
    test_set.to_csv(DATASET_DIR / 'test_trans.csv', index=False, sep=';')


train_transformer_pipeline()
