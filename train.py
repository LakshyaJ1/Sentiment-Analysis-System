"""
Movie Sentiment Analysis - Training Script
==========================================

This script trains a sentiment analysis model on movie review data.
Dataset: IMDB Dataset of 50K Movie Reviews from Kaggle
Place the CSV file as 'movie_reviews.csv' in the /data directory.
The CSV should have two columns: 'review' and 'sentiment'.
"""

import pandas as pd
import nltk
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class SentimentAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = LinearSVC(random_state=42)
    
    def preprocess_text(self, text):
        """
        Preprocess text data by:
        1. Converting to lowercase
        2. Removing HTML tags
        3. Removing punctuation and special characters
        4. Tokenizing
        5. Removing stopwords
        6. Applying Porter Stemming
        7. Joining tokens back to string
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
    
    def load_data(self, filepath):
        """Load movie reviews dataset"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
            return df
        except FileNotFoundError:
            print(f"Error: File '{filepath}' not found.")
            print("Please ensure you have downloaded the IMDB Dataset of 50K Movie Reviews")
            print("from Kaggle and placed it as 'movie_reviews.csv' in the /data directory.")
            return None
    
    def train(self, df):
        """Train the sentiment analysis model"""
        print("\nPreprocessing text data...")
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        
        # Prepare features and labels
        X = df['processed_review']
        y = df['sentiment']
        
        print("Extracting features using TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Train the model
        print("Training Linear SVM model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def save_models(self):
        """Save the trained model and vectorizer"""
        os.makedirs('saved_models', exist_ok=True)
        
        # Save the model
        joblib.dump(self.model, 'saved_models/sentiment_model.joblib')
        print("Model saved to: saved_models/sentiment_model.joblib")
        
        # Save the vectorizer
        joblib.dump(self.vectorizer, 'saved_models/tfidf_vectorizer.joblib')
        print("Vectorizer saved to: saved_models/tfidf_vectorizer.joblib")

def main():
    """Main training function"""
    print("Movie Sentiment Analysis - Training")
    print("=" * 40)
    
    # Initialize the analyzer
    analyzer = SentimentAnalyzer()
    
    # Load the dataset
    data_path = 'data/movie_reviews.csv'
    df = analyzer.load_data(data_path)
    
    if df is not None:
        # Train the model
        accuracy = analyzer.train(df)
        
        # Save the trained models
        analyzer.save_models()
        
        print(f"\nTraining completed successfully!")
        print(f"Final accuracy: {accuracy:.4f}")
        print("You can now use predict.py to make predictions on new reviews.")
    else:
        print("Training failed: Dataset not found.")

if __name__ == "__main__":
    main()
