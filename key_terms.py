# Write your code here
import string
from lxml import etree
import xml.etree.ElementTree as ET
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
stop_list = stopwords.words('english') + list(string.punctuation)
#### prepare dataset and list of headers
#tree = ET.parse('news.xml')
#root = tree.getroot()
root = etree.parse('news.xml').getroot()
corpus = root[0]
headers = []
dataset = []
for news in corpus:
    headers.append(news[0].text + ':') # save header
    text = news[1].text
    text_list = word_tokenize(text.lower())
    for i in range(len(text_list)):
        text_list[i] = lemmatizer.lemmatize(text_list[i]) # replace word by lemmatizered
    # Get rid of punctuation, stopwords, not-noun:
    only_noun_list = []
    for word in text_list:
        if word not in stop_list:
            if nltk.pos_tag([word])[0][1] == 'NN':
                only_noun_list.append(word)
    dataset.append(' '.join(only_noun_list)) #append only nouns from one news to dataset as string of words
####
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(dataset)
voc_terms = vectorizer.get_feature_names()
for i in range(len(dataset)):
    print(headers[i])
    vector = tfidf_matrix[i]
    df = pd.DataFrame(vector.T.todense(), index=voc_terms, columns=['tfidf'])
    df.index.name = 'word'
    df = df.sort_values(['tfidf', 'word'], ascending=[False, False]).head()
    print(' '.join(list(df.index)))
