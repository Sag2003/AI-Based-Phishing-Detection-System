from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize the Flask app
app = Flask(__name__)

# Load pretrained models and resources
def load_resources():
    global tfidf_vectorizer, models, lstm_model, tokenizer

    # Load TF-IDF vectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        tfidf_vectorizer = pickle.load(vectorizer_file)
    
    # Load pretrained models
    model_names = ["logistic_regression", "random_forest", "svm", "naive_bayes"]
    models = {}
    for model_name in model_names:
        with open(f'{model_name}_model.pkl', 'rb') as model_file:
            models[model_name] = pickle.load(model_file)
    
    # Load LSTM model
    lstm_model = load_model('lstm_model.h5')
    
    # Compile the LSTM model to eliminate the warning
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Load tokenizer
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    
    # Prediction using traditional models
    results = {}
    for model_name, model in models.items():
        prediction = model.predict(tfidf_vectorizer.transform([email_text]))
        results[model_name] = "Phishing" if prediction[0] == 1 else "Legitimate"
    
    # Prediction using LSTM
    email_seq = tokenizer.texts_to_sequences([email_text])
    email_lstm = pad_sequences(email_seq, padding='post', maxlen=1000)  # Use appropriate maxlen
    lstm_prediction = lstm_model.predict(email_lstm)
    results["LSTM"] = "Phishing" if lstm_prediction[0][0] > 0.5 else "Legitimate"

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)