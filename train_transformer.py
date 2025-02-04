import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.transforms import transform_remove_cid, transform_remove_mail_header
from preprocessing.transforms import transform_remove_md5_hash, get_all_body_transforms
from labels import get_id
from labels_manual_groups import get_id as man_id

val_size = 0.1


def apply_transformers_body(df, transformers):
    if len(transformers) == 0:
        return
    for transformer in transformers:
        df.MailTextBody = df['MailTextBody'].apply(lambda x: transformer(x))


def transform_mail_subject(x):
    if len(x) == 0:
        return 'no_content'
    return x


def concat(df):
    df['MailComplete'] = df['MailSubject'] + ' ' + df['MailTextBody']


def apply_transformers_subject(df):
    df.MailSubject = df.MailSubject.apply(lambda x: transform_mail_subject(str(x)))
    df.MailSubject = df.MailSubject.apply(lambda x: transform_remove_md5_hash(x))


def apply_label_transformer(df):
    df.ServiceProcessed = df['ServiceProcessed'].apply(lambda x: get_id(x))
    df.fillna("nothing", inplace=True)
    df.ManualGroups = df.ManualGroups.apply(lambda x: man_id(x))


def train_transformer_pipeline(data_dir, transforms=get_all_body_transforms()):
    train_set = pd.read_csv(data_dir / 'train.csv', delimiter=';')
    apply_transformers_subject(train_set)
    apply_label_transformer(train_set)
    if len(transforms) > 0:
        apply_transformers_body(train_set, transforms)
    concat(train_set)
    labels = train_set[['ServiceProcessed', 'ManualGroups']].copy()
    train_set.drop(columns=['Impact', 'Urgency', 'IncidentType', 'ServiceProcessed', 'ManualGroups'], inplace=True)
    train_set['ServiceProcessed'] = labels['ServiceProcessed']
    train_set['ManualGroups'] = labels['ManualGroups']
    train_set.drop(columns=['Unnamed: 0'], inplace=True)

    train_set.to_csv(data_dir / 'train_trans.csv', sep=';', index=False)
    train_set.to_json(data_dir / 'train_trans.json', orient='table')

    manual_set = train_set.dropna()
    manual_set.to_csv(data_dir / 'manual_trans.csv', sep=';', index=False)
    manual_set.to_json(data_dir / 'manual_trans.json', orient='table')
    #train_csv, validation_trans = train_test_split(train_set, test_size=val_size)
    #train_csv.to_csv(data_dir / 'train_trans.csv', sep=';', index=False)
    #train_csv.to_json(data_dir / 'train_trans.json', orient='table')
    # train_csv, validation_trans = train_test_split(train_set, test_size=val_size)
    # train_csv.to_csv(data_dir / 'train_trans.csv', sep=';', index=False)
    # train_csv.to_json(data_dir / 'train_trans.json', orient='table')

    # validation_trans.to_csv(data_dir / 'validation_trans.csv', sep=';', index=False)
    # validation_trans.to_json(data_dir / 'validation_trans.json', orient='table')

    test_set = pd.read_csv(data_dir / 'test_reduced.csv', delimiter=';')
    apply_transformers_subject(test_set)
    concat(test_set)
    if len(transforms) > 0:
        apply_transformers_body(test_set, transforms)
    test_set['ServiceProcessed'] = 0
    test_set['ManualGroups'] = 0
    test_set.drop(columns=['Unnamed: 0'], inplace=True)
    test_set.to_csv(data_dir / 'test_trans.csv', sep=';', index=False)
    test_set.to_json(data_dir / 'test_trans.json', orient='table')
