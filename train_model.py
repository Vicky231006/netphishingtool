import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import re
from bs4 import BeautifulSoup

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Phishing keywords (same as in app.py)
PHISHING_KEYWORDS = {
    'urgency': [
        'urgent', 'immediately', 'asap', 'now', 'today', 'important', 'verify',
        'suspended', 'restricted', 'limited', 'deadline', 'security', 'unusual',
        'suspicious', 'authenticate', 'validate'
    ],
    'reward': [
        'winner', 'won', 'winning', 'selected', 'congratulations', 'prize',
        'reward', 'gift', 'bonus', 'exclusive', 'instant', 'guaranteed'
    ],
    'threat': [
        'suspended', 'disabled', 'unauthorized', 'illegal', 'blocked', 'locked',
        'limited', 'deletion', 'terminate', 'cancel', 'expired', 'breach'
    ],
    'action': [
        'verify', 'confirm', 'validate', 'authenticate', 'click', 'login',
        'sign in', 'update', 'download', 'submit', 'fill', 'complete'
    ]
}

def preprocess_text(text):
    """Clean and preprocess the email text"""
    if isinstance(text, str):
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    return ''

def extract_manual_features(text):
    """Extract additional features from text"""
    text_lower = text.lower()
    
    features = {
        'length': len(text),
        'contains_urgency': any(word in text_lower for word in PHISHING_KEYWORDS['urgency']),
        'contains_money': any(word in text_lower for word in ['money', 'bank', 'account', 'payment']),
        'contains_suspicious': any(word in text_lower for word in PHISHING_KEYWORDS['action']),
        'url_count': text_lower.count('http'),
        'contains_greeting': any(word in text_lower for word in ['hi', 'hello', 'dear']),
    }
    
    return list(features.values())

def train_model(data_path):
    """Train the phishing detection model"""
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Ensure the dataset has the required columns
    required_columns = ['text_combined', 'label']  # Updated column names
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
    print("Preprocessing text...")
    # Preprocess the text
    df['processed_text'] = df['text_combined'].apply(preprocess_text)  # Updated column name
    
    # Labels are already binary (0 for legitimate, 1 for phishing)
    # No need to convert labels
    
    print("Extracting features...")
    # TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df['processed_text'])
    
    # Manual features
    X_manual = np.array([extract_manual_features(text) for text in df['processed_text']])
    
    # Combine features
    X = np.hstack([X_tfidf.toarray(), X_manual])
    y = df['label']
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save models and feature names
    print("\nSaving models...")
    joblib.dump(classifier, 'models/phishing_classifier.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # Save feature names (TF-IDF features + manual feature names)
    feature_names = vectorizer.get_feature_names_out().tolist() + [
        'length', 'contains_urgency', 'contains_money', 
        'contains_suspicious', 'url_count', 'contains_greeting'
    ]
    joblib.dump(feature_names, 'models/feature_names.joblib')
    
    print("Training complete! Models saved in 'models' directory.")

if __name__ == "__main__":
    # Specify your dataset path here
    data_path = "dataset/phishing_email.csv"
    train_model(data_path) 