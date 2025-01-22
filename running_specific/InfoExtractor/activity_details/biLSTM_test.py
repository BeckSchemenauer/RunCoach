from collections import Counter

import torch
import torch.nn as nn
import nltk
from sklearn.preprocessing import LabelEncoder
import json

nltk.download('punkt')


# Assuming the BiLSTMModel and tokenizer from the original code are present here

# Define the BiLSTM model architecture
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# Load the vocabulary and label encoder from the training script
with open('generated_activity_sentences.json', 'r') as file:
    data = json.load(file)

# Build vocabulary (same as the training script)
all_tokens = [token for item in data for token in nltk.word_tokenize(item['sentence'])]
vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}
vocab_size = len(vocab) + 1

label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

# Initialize the model with the same parameters as during training
embed_dim = 50
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)

# Load the saved model weights
model.load_state_dict(torch.load('bilstm_model.pth'))
model.eval()  # Set the model to evaluation mode


# Tokenizer function (same as in the training script)
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


# Function to predict tags for a sentence
def predict_tags(sentence):
    tokens = [vocab.get(token, 0) for token in tokenizer(sentence)]  # Convert tokens to indices
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    outputs = model(tokens_tensor)
    predictions = torch.argmax(outputs, dim=2)  # Get predicted labels for each token
    predicted_labels = label_encoder.inverse_transform(predictions.squeeze().numpy())
    return predicted_labels


# Create a loop to take user input and print predicted tags
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    predicted_tags = predict_tags(user_input)
    print(f"Predicted tags: {predicted_tags}")
