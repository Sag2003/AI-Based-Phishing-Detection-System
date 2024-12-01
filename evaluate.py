import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import pickle
from keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100 * ((1000 - 2) // 2), 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
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

# Load test data
def load_test_data(batch_size=32):
    df = pd.read_csv('Ling.csv')
    sequences = tokenizer.texts_to_sequences(df['body'].tolist())
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=1000)
    labels = df['label'].values

    # Convert to tensors
    data = torch.tensor(padded_sequences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 64

# Initialize models
cnn_model = CNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
cnn_model.load_state_dict(torch.load('cnn_model.pth'))
cnn_model.eval()

rnn_model = RNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
rnn_model.load_state_dict(torch.load('rnn_model.pth'))
rnn_model.eval()

lstm_model = LSTMModel(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
lstm_model.load_state_dict(torch.load('lstm_model.pth'))
lstm_model.eval()

# Evaluate the models
def evaluate_model(model, dataloader):
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = (outputs.cpu().numpy() > 0.5).astype(int).flatten()
            predictions.extend(preds)
            true_labels.extend(labels.numpy())

            # Clear unused memory
            torch.cuda.empty_cache()

    return np.array(predictions), np.array(true_labels)

# Load test data in batches
test_loader = load_test_data(batch_size=32)

# Evaluate each model
print("Evaluating CNN model...")
cnn_preds, test_labels = evaluate_model(cnn_model, test_loader)
cnn_accuracy = accuracy_score(test_labels, cnn_preds)

print("Evaluating RNN model...")
rnn_preds, _ = evaluate_model(rnn_model, test_loader)
rnn_accuracy = accuracy_score(test_labels, rnn_preds)

print("Evaluating LSTM model...")
lstm_preds, _ = evaluate_model(lstm_model, test_loader)
lstm_accuracy = accuracy_score(test_labels, lstm_preds)

# Print results
print(f"CNN Model Accuracy: {cnn_accuracy:.4f}")
print(f"RNN Model Accuracy: {rnn_accuracy:.4f}")
print(f"LSTM Model Accuracy: {lstm_accuracy:.4f}")

print("\nCNN Classification Report:\n", classification_report(test_labels, cnn_preds))
print("\nRNN Classification Report:\n", classification_report(test_labels, rnn_preds))
print("\nLSTM Classification Report:\n", classification_report(test_labels, lstm_preds))