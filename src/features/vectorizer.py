from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def vectorize_text(df, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['text'])
    return X, tfidf

def save_vectorizer(tfidf, path):
    with open(path, 'wb') as f:
        pickle.dump(tfidf, f)
