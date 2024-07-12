# Sentiment Analysis Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NLTK](https://img.shields.io/badge/NLTK-3.5%2B-green)
![spaCy](https://img.shields.io/badge/spaCy-3.0%2B-yellow)
![TextBlob](https://img.shields.io/badge/TextBlob-0.15%2B-orange)

## Overview

Welcome to the Sentiment Analysis Project! This project aims to create a robust sentiment analysis tool that can determine the emotional tone of a given text—whether it is positive, negative, or neutral. The project leverages popular Python libraries such as NLTK, spaCy, and TextBlob to preprocess text, analyze sentiment, and classify it into a 5-star rating system.

## Features

- **Text Preprocessing:** Clean and preprocess text data to remove HTML tags, convert to lowercase, remove punctuation, and perform tokenization and lemmatization.
- **Sentiment Analysis:** Use TextBlob to analyze the sentiment of the text and classify it as positive, negative, or neutral.
- **5-Star Rating System:** Convert sentiment labels into a user-friendly 5-star rating system.
- **Model Persistence:** Check if a pre-trained model exists to avoid retraining, saving time on subsequent runs.
- **Interactive Predictions:** Test the model with custom sentences to predict their sentiment and corresponding star rating.

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- NLTK
- spaCy
- TextBlob
- Scikit-learn
- Joblib
- Pandas

### Dataset

The dataset used in this project is the IMDb Movie Reviews Dataset, which contains 50,000 movie reviews labeled as positive or negative. This dataset is commonly used for binary sentiment classification and provides a balanced set of reviews with 25,000 positive and 25,000 negative examples.

## Acknowledgments

- [NLTK](https://www.nltk.org/)
- [spaCy](https://spacy.io/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/)
- [Scikit-learn](https://scikit-learn.org/stable/)

 
