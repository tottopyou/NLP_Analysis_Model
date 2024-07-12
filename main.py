from textblob import TextBlob
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


df = pd.read_csv('dataset/IMDB Dataset.csv')

df = df[:10000]

print(df)

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Assuming the review column is named 'review'
df['cleaned_text'] = df['review'].apply(preprocess_text)

df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

# Classify the sentiment
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment_label'], test_size=0.2, random_state=42)

# Vectorize the text data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Start of training")

# Train a classifier
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Save the trained model and vectorizer
joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

def predict_sentiment(sentence):
    # Load the saved model and vectorizer
    clf = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    # Preprocess the input sentence
    cleaned_sentence = preprocess_text(sentence)
    # Vectorize the cleaned sentence
    sentence_vec = vectorizer.transform([cleaned_sentence])
    # Predict sentiment
    sentiment_label = clf.predict(sentence_vec)[0]
    return sentiment_label

def sentiment_to_stars(sentiment_label):
    if sentiment_label == "positive":
        return 5
    elif sentiment_label == "neutral":
        return 3
    else:
        return 1

# Test the model with a new sentence
new_sentence = "I love this movie! It's amazing and full of great performances."
predicted_sentiment = predict_sentiment(new_sentence)
stars = sentiment_to_stars(predicted_sentiment)
print(f"The sentiment of the sentence '{new_sentence}' is {predicted_sentiment}.", "\n",
      f"In 5 stars it will be {stars}")
