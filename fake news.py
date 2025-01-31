import pandas as pd
import numpy as np
import re
import string
import nltk
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
from flask_cors import CORS

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset (Assuming a CSV file with 'text' and 'label' columns)
df = pd.read_csv("fake_news_dataset.csv")

# Data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    text = re.sub("\d+", "", text)  # Remove numbers
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])
    return text

df['text'] = df['text'].apply(preprocess_text)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Building the model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)

