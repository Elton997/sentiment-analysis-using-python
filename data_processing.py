import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the sentiment data
data = pd.read_csv('sentiment_data.csv')

# Preprocessing function
def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    data['processed_text'] = data['text'].apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word.isalnum() and word not in stop_words]))
    return data

# Vectorization function
def vectorize_data(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_text'])
    return X, vectorizer

# Preprocess the data
processed_data = preprocess_data(data)

# Vectorize the data
X, vectorizer = vectorize_data(processed_data)

# Save vectorized data
pd.DataFrame(X.toarray()).to_csv('vectorized_data.csv', index=False)

# Save labels
processed_data['sentiment'].to_csv('labels.csv', index=False, header=['sentiment'])

# Save vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
