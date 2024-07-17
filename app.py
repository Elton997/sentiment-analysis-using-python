from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.json
    text = data['text']
    vectorized_text = vectorizer.transform([text])
    sentiment = model.predict(vectorized_text)
    return jsonify({'sentiment': sentiment[0]})

if __name__ == '__main__':
    app.run(debug=True)
