import pandas as pd
from sklearn.model_selection import train_test_split
from transfomers import transform_remove_cid, transform_remove_mail_header

val_size = 0.1

def apply_transformers(df, transformers):
    if len(transformers) == 0:
        return
    for transformer in transformers:
        df.MailTextBody = df['MailTextBody'].apply(lambda x: transformer(x))


def train_transformer_pipeline(data_dir, transformers=[]):
    train_set = pd.read_csv(data_dir / 'train.csv', delimiter=';')
    if len(transfomers > 0):
        apply_transformers(train_set, transformers)
    labels = train_set[['ServiceProcessed', 'ManualGroups']].copy()
    train_set.drop(columns=['Impact', 'Urgency', 'IncidentType', 'ServiceProcessed', 'ManualGroups'], inplace=True)
    train_set['ServiceProcessed'] = labels['ServiceProcessed']
    train_set['ManualGroups'] = labels['ManualGroups']
    train_set.rename(columns={'Unnamed: 0': ''}, inplace=True)

    train_csv, validation_trans = train_test_split(train_set, test_size=val_size)
    train_csv.to_csv(data_dir / 'train_trans.csv', sep=';', index=False)
    validation_trans.to_csv(data_dir / 'validation_trans.csv', sep=';', index=False)

    test_set = pd.read_csv(data_dir / 'test_reduced.csv', delimiter=';')
    if len(transfomers > 0):
        apply_transformers(test_set, transformers)
    test_set['ServiceProcessed'] = ' '
    test_set['ManualGroups'] = ' '
    test_set.rename(columns={'Unnamed: 0': ''}, inplace=True)
    test_set.to_csv(data_dir / 'test_trans.csv', index=False, sep=';')
