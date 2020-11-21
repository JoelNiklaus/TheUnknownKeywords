# TheUnknownKeywords
Repo for the Hackathon Sk[AI] is the limit

## Task 1 Predictions

We tried several BERT Models for our Predictions. Our bests results were with dbmdz/bert-base-german-uncased from huggingface.co

Our Pipeline is:

1. Clean the data:
   1. Remove some JS and HMTL Tagging
   2. Remove Mail Headers from AW and FW Mails
2. Train Model with full set, there might be an overfit
3. Predict classes with trained model

Using BERT pre-trained `dbmdz/bert-base-german-uncased` model.

Some results:

| Checkpoint | F1-Score |
|------------|----------|
| 1750       | 0.733    |
| 2250       | 0.726    |
| 6750       | 0.714    |

Corpus preprocessing:

* Footers and signatures were removed
* Email subject was concatenated with the email body
* Long emails were split into smaller chunks and duplicated rows were inserted
* HTML and JS code was removed


## Task 2 Service Merging
