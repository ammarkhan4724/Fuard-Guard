import numpy as np
import pandas as pd 
import warnings
from nltk.corpus import stopwords
import string, nltk
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



def preprocess(text):
    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])

def stem_words(text):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def lemmatize_words(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def purify_data():
    try:
        df = pd.read_csv('fake reviews dataset.csv')
        df.isnull().sum()
        df['text_'] = df['text_'].apply(lambda x: stem_words(x))
        df['text_'] = df['text_'].astype(str)
        preprocess(df['text_'][4])
        df['text_'][:10000] = df['text_'][:10000].apply(preprocess)
        df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)
        df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)
        df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)
        df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)
        df['text_'] = df['text_'].str.lower()
        df["text_"] = df["text_"].apply(lambda text: lemmatize_words(text))
        df.to_csv('Preprocessed Fake Reviews Detection Dataset.csv')
    except Exception as e:
        print(e)
        print("We need a file named as 'fake reviews dataset.csv' as dataset. Please provide it first")



warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('stopwords')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

purify_data()