import spacy
import pandas as pd
import collections
import en_core_web_lg
import pickle
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter
from newsapi import NewsApiClient
nltk.download('stopwords')

nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient(api_key='d376fc1d36904c7a801fa1da2b6f23f6')
feed = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-27', to='2022-03-27', sort_by='relevancy', page_size=100)
articles = feed['articles']
archive = []
filename = 'covid_feed.pckl'
pickle.dump(feed, open(filename, 'wb'))
filename = 'covid_feed.pckl'
loaded_model = pickle.load(open(filename, 'rb'))
filepath = 'covid_feed.pckl'
pickle.dump(loaded_model, open(filepath, 'wb'))

for i, article in enumerate(articles):
    archive.append({'title': article['title'], 'date': article['publishedAt'], 'desc': article['description'],
                    'content': article['content']})
df = pd.DataFrame(archive)
df = df.dropna()  # Drops any record with NA or missing value
# print(df.head(50))
tokenizer = RegexpTokenizer(r'\w+')

def get_keywords_eng(token):
    result = []
    stop_words = stopwords.words('english')
    for i in token:
        if (i in stop_words):
            continue
        else:
            result.append(i)

    return result


results = []
for content in df.content.values:
    content = tokenizer.tokenize(content)
    results.append([x[0] for x in Counter(get_keywords_eng(content)).most_common(5)])
df['keywords'] = results

print(df.head(50))