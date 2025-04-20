from flask import Flask, request, jsonify
from flask_cors import CORS
import re
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from urllib.parse import urlparse
import requests
from collections import Counter
import string

app = Flask(__name__)
CORS(app)

# Load ML models if they exist
MODEL_PATH = 'models/phishing_classifier.joblib'
VECTORIZER_PATH = 'models/tfidf_vectorizer.joblib'
FEATURE_NAMES_PATH = 'models/feature_names.joblib'

try:
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    ml_model_available = True
    print("ML model loaded successfully!")
except Exception as e:
    ml_model_available = False
    print(f"Error loading ML model: {str(e)}")
    print("Falling back to rule-based analysis")

# Enhanced phishing patterns
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

# Common legitimate email domains
LEGITIMATE_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
    'icloud.com', 'protonmail.com', 'mail.com'
}

def check_url_reputation(url):
    """Check URL reputation using basic heuristics"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Check for IP addresses in URL
        if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
            return 1.0
        
        # Check for suspicious TLDs
        suspicious_tlds = {'.xyz', '.top', '.work', '.loan', '.click', '.party'}
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            return 0.8
            
        # Check for character substitution (e.g., paypa1.com)
        common_substitutions = {
            'paypal': r'p[a@]yp[a@]l',
            'microsoft': r'm[i1]cr[o0]s[o0]ft',
            'apple': r'[a@]ppl[e3]',
            'amazon': r'[a@]m[a@]z[o0]n'
        }
        
        for brand, pattern in common_substitutions.items():
            if re.search(pattern, domain.lower()):
                return 0.9
                
        return 0.0
    except:
        return 0.5

def extract_ml_features(text):
    """Extract features for ML model"""
    # TF-IDF features
    X_text = vectorizer.transform([text])
    
    # Additional manual features
    manual_features = {
        'length': len(text),
        'contains_urgency': any(word in text.lower() for word in PHISHING_KEYWORDS['urgency']),
        'contains_money': any(word in text.lower() for word in ['money', 'bank', 'account', 'payment']),
        'contains_suspicious': any(word in text.lower() for word in PHISHING_KEYWORDS['action']),
        'url_count': text.lower().count('http'),
        'contains_greeting': any(word in text.lower() for word in ['hi', 'hello', 'dear']),
    }
    
    manual_feature_array = np.array([[v for v in manual_features.values()]])
    return np.hstack([X_text.toarray(), manual_feature_array])

def extract_rule_based_features(text):
    """Extract features using rule-based approach"""
    text_lower = text.lower()
    words = text_lower.split()
    
    features = {
        'length': len(text),
        'word_count': len(words),
        'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text),
        'contains_attachments': bool(re.search(r'attachment|attach|file', text_lower)),
        'contains_credentials': bool(re.search(r'password|login|credentials|account', text_lower)),
        'contains_personal_info': bool(re.search(r'social security|ssn|credit card|bank account', text_lower))
    }
    
    # Count occurrences of phishing keywords by category
    for category, keywords in PHISHING_KEYWORDS.items():
        features[f'{category}_words'] = sum(1 for word in words if word in keywords)
    
    # URL analysis
    features['suspicious_urls'] = []
    features['url_reputation_score'] = 0
    if features['urls']:
        url_scores = [check_url_reputation(url) for url in features['urls']]
        features['url_reputation_score'] = max(url_scores)
        features['suspicious_urls'] = [url for url, score in zip(features['urls'], url_scores) if score > 0.5]
    
    # Check for mixed case words (potential spoofing)
    features['mixed_case_words'] = len([w for w in text.split() if any(c.isupper() for c in w[1:]) and not w.isupper()])
    
    # Check for excessive punctuation
    features['excessive_punctuation'] = len([c for c in text if c in '!?*$#@']) > 3
    
    # Check for poor grammar/spelling (basic check for repeated words)
    word_counts = Counter(words)
    features['repeated_words'] = any(count > 3 for count in word_counts.values())
    
    return features

def calculate_rule_based_score(features):
    """Calculate risk score based on rule-based features"""
    score = 0
    
    # URL-based scoring
    if features['url_reputation_score'] > 0:
        score += features['url_reputation_score'] * 0.3
    
    # Keyword-based scoring
    keyword_score = sum([
        features.get('urgency_words', 0) * 0.15,
        features.get('reward_words', 0) * 0.15,
        features.get('threat_words', 0) * 0.2,
        features.get('action_words', 0) * 0.15
    ])
    score += min(keyword_score, 0.4)
    
    # Other suspicious elements
    if features['contains_credentials']:
        score += 0.15
    if features['contains_personal_info']:
        score += 0.15
    if features['mixed_case_words'] > 2:
        score += 0.1
    if features['excessive_punctuation']:
        score += 0.05
    if features['repeated_words']:
        score += 0.05
    
    return min(score, 1.0)

def analyze_email(email_text):
    """Analyze email content using both ML and rule-based approaches"""
    # Get rule-based features and score
    rule_features = extract_rule_based_features(email_text)
    rule_score = calculate_rule_based_score(rule_features)
    
    # Get ML-based score if available
    if ml_model_available:
        try:
            X = extract_ml_features(email_text)
            ml_score = classifier.predict_proba(X)[0][1]  # Probability of phishing
        except:
            ml_score = 0.5  # Default to uncertain if ML fails
    else:
        ml_score = rule_score
    
    # Combine scores (giving more weight to ML when available)
    if ml_model_available:
        final_score = 0.7 * ml_score + 0.3 * rule_score
    else:
        final_score = rule_score
    
    # Determine status
    if final_score > 0.7:
        status = "Phishing"
    elif final_score > 0.4:
        status = "Suspicious"
    else:
        status = "Safe"
    
    # Generate highlights
    highlights = []
    
    # Add ML confidence if available
    if ml_model_available:
        highlights.append(f"ML Model Confidence: {ml_score*100:.1f}%")
    
    # Add rule-based findings
    for category in PHISHING_KEYWORDS:
        if rule_features.get(f'{category}_words', 0) > 0:
            highlights.append(f"Found {rule_features[f'{category}_words']} {category}-related words")
    
    if rule_features['suspicious_urls']:
        highlights.append(f"Detected {len(rule_features['suspicious_urls'])} suspicious URLs")
    if rule_features['contains_credentials']:
        highlights.append("Requests login credentials")
    if rule_features['contains_personal_info']:
        highlights.append("Requests sensitive personal information")
    
    return {
        "status": status,
        "risk_score": final_score,
        "ml_score": ml_score if ml_model_available else None,
        "rule_score": rule_score,
        "features": rule_features,
        "highlights": highlights
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    email_text = data.get('email_text', '')
    
    if not email_text:
        return jsonify({"error": "No email text provided"}), 400
    
    result = analyze_email(email_text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 