# Phishing Email Detector

A machine learning and rule-based system for detecting phishing emails.

## Features

- Machine learning model using RandomForestClassifier
- Rule-based analysis for additional security
- REST API endpoint for email analysis
- Comprehensive feature extraction
- URL reputation checking

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the model files (see below)
4. Run the Flask application:
   ```bash
   python app.py
   ```

## Model Files

The model files are not included in the repository due to their size. You need to download them separately:

1. `models/phishing_classifier.joblib`
2. `models/tfidf_vectorizer.joblib`
3. `models/feature_names.joblib`

## API Usage

Send a POST request to `/analyze` with the email text:

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Your email content here"}'
```

## Response Format

```json
{
  "status": "Phishing|Suspicious|Safe",
  "risk_score": 0.0-1.0,
  "ml_score": 0.0-1.0,
  "rule_score": 0.0-1.0,
  "features": {...},
  "highlights": [...]
}
```

## License

MIT License 