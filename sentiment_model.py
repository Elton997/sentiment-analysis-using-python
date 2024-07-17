from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib

def train_model(vectorized_data_path, labels_path):
    X = pd.read_csv(vectorized_data_path)
    y = pd.read_csv(labels_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train.values.ravel())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy}')

    joblib.dump(model, 'sentiment_model.pkl')

if __name__ == "__main__":
    train_model('vectorized_data.csv', 'labels.csv')
