from textblob import TextBlob
import re
import nltk
import joblib



def preprocess_text(text,stop_words,nlp):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    doc = nlp(" ".join(tokens))
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def predict_sentiment(sentence,stop_words,nlp):
    # Load the saved model and vectorizer
    clf = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    # Preprocess the input sentence
    cleaned_sentence = preprocess_text(sentence,stop_words, nlp)
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