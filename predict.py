"""
Movie Sentiment Analysis - Prediction Script
============================================

This script loads the trained sentiment analysis model and makes predictions
on new movie review text.
"""

import joblib
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class SentimentPredictor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            self.model = joblib.load('saved_models/sentiment_model.joblib')
            self.vectorizer = joblib.load('saved_models/tfidf_vectorizer.joblib')
            print("Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}")
            print("Please run train.py first to train and save the models.")
            return False
        return True
    
    def preprocess_text(self, text):
        """
        Preprocess text data using the same steps as in training:
        1. Convert to lowercase
        2. Remove HTML tags
        3. Remove punctuation and special characters
        4. Tokenize
        5. Remove stopwords
        6. Apply Porter Stemming
        7. Join tokens back to string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove punctuation and special characters, keep only alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                stemmed_token = self.stemmer.stem(token)
                processed_tokens.append(stemmed_token)
        
        # Join tokens back to string
        return ' '.join(processed_tokens)
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a given movie review text.
        
        Args:
            text (str): Raw movie review text
            
        Returns:
            str: 'positive' or 'negative'
        """
        if self.model is None or self.vectorizer is None:
            return "Error: Models not loaded"
        
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Transform using the fitted vectorizer
        text_tfidf = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.model.predict(text_tfidf)[0]
        
        return prediction

def main():
    """Example usage of the sentiment predictor"""
    print("Movie Sentiment Analysis - Prediction")
    print("=" * 40)
    
    # Initialize the predictor
    predictor = SentimentPredictor()
    
    # Example positive review
    positive_review = """
    This movie was absolutely fantastic! The acting was superb, the plot was 
    engaging, and the cinematography was breathtaking. I was on the edge of my 
    seat the entire time. The character development was excellent and the ending 
    was very satisfying. I would definitely recommend this movie to anyone who 
    enjoys great storytelling and brilliant performances. A masterpiece!
    """
    
    # Example negative review
    negative_review = """
    This was one of the worst movies I've ever seen. The plot made no sense, 
    the acting was terrible, and the dialogue was cringe-worthy. I couldn't 
    wait for it to end. The special effects looked cheap and the whole thing 
    felt like a waste of time and money. I would not recommend this movie to 
    anyone. Absolutely disappointing!
    """
    
    # Make predictions
    print("\nExample Predictions:")
    print("-" * 20)
    
    result1 = predictor.predict_sentiment(positive_review)
    print(f"Positive Review Prediction: {result1}")
    
    result2 = predictor.predict_sentiment(negative_review)
    print(f"Negative Review Prediction: {result2}")
    
    # Interactive prediction mode
    print("\nInteractive Mode:")
    print("Enter a movie review to analyze (or 'quit' to exit):")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input:
            prediction = predictor.predict_sentiment(user_input)
            print(f"Predicted sentiment: {prediction}")
        else:
            print("Please enter a valid review.")

if __name__ == "__main__":
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading required NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    
    main()
