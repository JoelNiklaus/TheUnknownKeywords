import pandas as pd
def train_transformer_pipeline():
    train_set = pd.read_csv('train.csv', delimiter=';')
    labels = train_set[['ServiceProcessed', 'ManualGroups']].copy()
    train_set.drop(columns=['Impact','Urgency','IncidentType', 'ServiceProcessed', 'ManualGroups'], inplace=True)
    train_set['ServiceProcessed'] = labels['ServiceProcessed']
    train_set['ManualGroups'] = labels['ManualGroups']
    train_set.rename(columns={'Unnamed: 0':''}, inplace=True)
    train_set.to_csv('train_trans.csv', sep=';', index=False)
train_transformer_pipeline()