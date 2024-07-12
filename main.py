import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os
from functions import sentiment_to_stars, predict_sentiment,preprocess_text,get_sentiment


model_path = 'sentiment_model.pkl'
vectorizer_path = 'vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):

    df = pd.read_csv('dataset/IMDB Dataset.csv')

    df = df[:200]

    print(df.head())
    print(df.columns)

    nltk.download('stopwords')
    nltk.download('punkt')

    stop_words = set(stopwords.words('english'))
    nlp = spacy.load("en_core_web_sm")

    df['cleaned_text'] = df['review'].apply(preprocess_text, stop_words, nlp)

    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment_label'], test_size=0.2, random_state=42)

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Start of training")

    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)

    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

else:
    print("Model and vectorizer already exist. Loading them...")
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load("en_core_web_sm")

new_sentence = "I love this movie! It's amazing and full of great performances."
predicted_sentiment = predict_sentiment(new_sentence, stop_words, nlp)
stars = sentiment_to_stars(predicted_sentiment)
print(f"The sentiment of the sentence '{new_sentence}' is {predicted_sentiment}.", "\n",
      f"In 5 stars it will be {stars}")
