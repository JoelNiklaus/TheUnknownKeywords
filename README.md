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

## Task 2 Service Merging


