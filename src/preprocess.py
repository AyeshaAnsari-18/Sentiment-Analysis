import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

def prepare_data(data):
    column_names = ['userId', 'productId', 'Rating', 'timestamp']

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        df.columns = column_names
    else:
        df = pd.read_csv(data, header=None, names=column_names)

    df.dropna(inplace=True)

    df['sentiment'] = df['Rating'].apply(lambda x: 'positive' if x >= 4 else 'negative')

    df['reviewText'] = df['sentiment'].apply(
        lambda x: "I love this product!" if x == 'positive' else "I hate this product.")

    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['sentiment'])

    return df, le

def vectorize_data(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

    X = tfidf.fit_transform(df['reviewText'])
    y = df['label_encoded']
    
    return train_test_split(X, y, test_size=0.2, random_state=42), tfidf
