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
max_length = 1000  # Set this according to your model's expected input length

# VirusTotal API key
VIRUSTOTAL_API_KEY = 'Key Here'  # Replace with your VirusTotal API key

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
        x = x.permute(0, 2, 1)  # Change to (batch_size, embedding_dim, seq_length)
        x = self.conv1(x)
        x = self.pool(x)

        # Flatten the output
        x = x.view(x.size(0), -1)
        output_size = x.size(1)

        # Adjust fully connected layer size if needed
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

def check_virustotal_links(email_text):
    # Extract links from the email text
    links = re.findall(r'(https?://\S+)', email_text)
    for link in links:
        # Encode the link for VirusTotal API
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

    # Preprocess the input for the models
    email_seq = tokenizer.texts_to_sequences([email_text])
    email_padded = tf.keras.preprocessing.sequence.pad_sequences(email_seq, padding='post', maxlen=max_length)
    email_tensor = torch.tensor(email_padded, dtype=torch.long).to(device)

    # VirusTotal Link Check
    virustotal_result = check_virustotal_links(email_text)
    
    # If VirusTotal flags the link as malicious/phishing, force the Final Verdict to Phishing
    if virustotal_result == "VirusTotal: Malicious/Phishing":
        final_verdict = "Phishing"
        lstm_result = "Phishing"
        cnn_result = "Phishing"
        rnn_result = "Phishing"
    else:
        # LSTM Prediction
        lstm_prediction = lstm_model(email_tensor)
        lstm_result = "Phishing" if lstm_prediction.item() > 0.5 else "Legitimate"

        # CNN Prediction
        if email_tensor.size(1) < 3:  # Less than kernel size
            cnn_result = "Input too short for CNN"
        else:
            cnn_prediction = cnn_model(email_tensor)
            cnn_result = "Phishing" if cnn_prediction.item() > 0.5 else "Legitimate"

        # RNN Prediction
        rnn_prediction = rnn_model(email_tensor)
        rnn_result = "Phishing" if rnn_prediction.item() > 0.5 else "Legitimate"

        # Determine final verdict based on majority voting
        predictions = [lstm_result, cnn_result, rnn_result]
        final_verdict = "Phishing" if predictions.count("Phishing") > predictions.count("Legitimate") else "Legitimate"

    # Prepare the results dictionary with Final Verdict only once
    results = {
        "VirusTotal": virustotal_result,
        "LSTM": lstm_result,
        "CNN": cnn_result,
        "RNN": rnn_result
    }

    # Add Final Verdict if not already set by VirusTotal
    if virustotal_result != "VirusTotal: Malicious/Phishing":
        results["Final Verdict"] = final_verdict

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
