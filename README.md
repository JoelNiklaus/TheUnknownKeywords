# TheUnknownKeywords
Repo for the Hackathon Sk[AI] is the limit

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

