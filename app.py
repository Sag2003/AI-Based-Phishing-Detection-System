from flask import Flask, request, jsonify, render_template 
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import tensorflow as tf
import re
import requests
import base64

# Check for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define maximum length for inputs
max_length = 1000  

# VirusTotal API key
VIRUSTOTAL_API_KEY = '18808001f13a70f6395c527d42962e7c0542c6b05e6513f28c931aea88dcb7ae'  

# Define CNN Model using PyTorch
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100 * (max_length // 2 - 1), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        output_size = x.size(1)
        if output_size != self.fc.in_features:
            self.fc = nn.Linear(output_size, 1)
        x = torch.sigmoid(self.fc(x))
        return x

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = torch.sigmoid(self.fc(lstm_out))
        return x

# Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        rnn_out, _ = self.rnn(x)
        rnn_out = rnn_out[:, -1, :]
        x = torch.sigmoid(self.fc(rnn_out))
        return x

# Load and prepare the models
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 64

cnn_model = CNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
cnn_model.load_state_dict(torch.load('cnn_model.pth', map_location=device, weights_only=True))
cnn_model.eval()

lstm_model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
lstm_model.load_state_dict(torch.load('lstm_model.pth', map_location=device, weights_only=True))
lstm_model.eval()

rnn_model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
rnn_model.load_state_dict(torch.load('rnn_model.pth', map_location=device, weights_only=True))
rnn_model.eval()

app = Flask(__name__)

# Function to check VirusTotal for malicious links
def check_virustotal_links(email_text):
    links = re.findall(r'(https?://\S+)', email_text)
    for link in links:
        encoded_link = base64.urlsafe_b64encode(link.encode()).decode().strip('=')
        url = f"https://www.virustotal.com/api/v3/urls/{encoded_link}"
        headers = {'x-apikey': VIRUSTOTAL_API_KEY}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            analysis_results = data.get("data", {}).get("attributes", {}).get("last_analysis_results", {})
            for result in analysis_results.values():
                if result.get("category") in ["malicious", "phishing"]:
                    return "VirusTotal: Malicious/Phishing"
    return "VirusTotal: Clean"

# Function to detect fake login forms
def detect_fake_form(email_text):
    fake_form_patterns = [
        r'<form.*?action=["\'].*?login.*?["\']',
        r'input.*?type=["\']password["\']',
        r'input.*?name=["\']user["\']',
        r'input.*?name=["\']pass["\']'
    ]
    for pattern in fake_form_patterns:
        if re.search(pattern, email_text, re.IGNORECASE):
            return "Fake Login Detected: Phishing"
    return "Fake Login: Clean"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email']
    
    virustotal_result = check_virustotal_links(email_text)
    fake_form_result = detect_fake_form(email_text)
    
    explanation = []
    if virustotal_result == "VirusTotal: Malicious/Phishing":
        explanation.append("One or more links in this email have been flagged as phishing or malicious by VirusTotal.")
    if fake_form_result == "Fake Login Detected: Phishing":
        explanation.append("The email contains a suspicious login form that may be used for phishing.")
    
    email_seq = tokenizer.texts_to_sequences([email_text])
    email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_seq, padding='post', maxlen=max_length)
    email_tensor = torch.tensor(email_padded, dtype=torch.long).to(device)
    
    lstm_prediction = lstm_model(email_tensor)
    lstm_result = "Phishing" if lstm_prediction.item() > 0.5 else "Legitimate"
    if lstm_result == "Phishing":
        explanation.append("The models have the detected patterns commonly associated with phishing emails.")
    
    final_verdict = "Phishing" if explanation else "Legitimate"
    results = {"Final Verdict": final_verdict, "Explanation": " ".join(explanation) if explanation else "No phishing indicators detected."}
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
