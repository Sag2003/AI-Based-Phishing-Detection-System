import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define device for CUDA
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load preprocessed data
df = pd.read_csv('Phishing_Email.csv', low_memory=False)

# Ensure the data is not empty
if df.empty or 'body' not in df.columns or 'label' not in df.columns:
    raise ValueError("Dataset is empty or missing required columns. Ensure the 'body' and 'label' columns exist.")

# Prepare inputs and labels
X_texts = df['body'].values
y = df['label'].astype(int).values

# Initialize the Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_texts)

# Convert texts to sequences
X_seq = tokenizer.texts_to_sequences(X_texts)
X_padded = pad_sequences(X_seq, padding='post', maxlen=1000)

# Split the data
if len(X_padded) == 0 or len(y) == 0:
    raise ValueError("Input data or labels are empty after preprocessing.")

X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define dataset class
class EmailDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)

# Define models
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100 * ((1000 - 3 + 1) // 2), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        x = torch.sigmoid(self.fc(h_n[-1]))
        return x

class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h_n = self.rnn(x)
        x = torch.sigmoid(self.fc(h_n[-1]))
        return x

# Initialize models
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 64

cnn_model = CNNModel(vocab_size, embedding_dim).to(device)
lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim).to(device)
rnn_model = RNNModel(vocab_size, embedding_dim, hidden_dim).to(device)

# Define loss and optimizers
criterion = nn.BCELoss()
cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)

# Training function
def train_model(model, optimizer, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Train and save models
train_loader = DataLoader(EmailDataset(X_train, y_train), batch_size=32, shuffle=True)

print("Training CNN model...")
train_model(cnn_model, cnn_optimizer, train_loader)
torch.save(cnn_model.state_dict(), 'cnn_model.pth')

print("Training LSTM model...")
train_model(lstm_model, lstm_optimizer, train_loader)
torch.save(lstm_model.state_dict(), 'lstm_model.pth')

print("Training RNN model...")
train_model(rnn_model, rnn_optimizer, train_loader)
torch.save(rnn_model.state_dict(), 'rnn_model.pth')

# Save tokenizer
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

print("Models and tokenizer saved successfully.")