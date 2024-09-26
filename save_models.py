import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and prepare data
df = pd.read_csv('Phishing_Email.csv', low_memory=False)
df = df[['body', 'label']].dropna()

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(df['body'])
y = df['label'].astype(int)

# Save TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf_vectorizer, vectorizer_file)

# Train and save traditional models
models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC(probability=True),
    "naive_bayes": MultinomialNB()
}

for model_name, model in models.items():
    model.fit(X, y)
    with open(f'{model_name}_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

# Prepare and train LSTM model
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['body'])
X_seq = tokenizer.texts_to_sequences(df['body'])
X_lstm = pad_sequences(X_seq, padding='post')
y_lstm = np.array(y)

# Save tokenizer
with open('tokenizer.pkl', 'wb') as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

def create_lstm_model(vocabulary_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size, output_dim=128))
    model.add(LSTM(128))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = create_lstm_model(len(tokenizer.word_index) + 1)
lstm_model.fit(X_lstm, y_lstm, epochs=5, verbose=1)
lstm_model.save('lstm_model.h5')