import torch
import torch.nn as nn
import nltk
from sklearn.preprocessing import LabelEncoder
import json
from collections import Counter

nltk.download('punkt')


# BiLSTM Model Definition
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


# Load vocabulary and label encoder
with open('test_sentences.json', 'r') as file:
    data = json.load(file)

all_tokens = [token for item in data for token in nltk.word_tokenize(item['sentence'])]
vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}
vocab_size = len(vocab) + 1

label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

# Initialize and load model
embed_dim = 50
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('bilstm_model.pth'))
model.eval()


# Tokenizer function
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


# Predict tags for a sentence
def predict_tags(sentence):
    tokens = [vocab.get(token, 0) for token in tokenizer(sentence)]
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)
    outputs = model(tokens_tensor)
    predictions = torch.argmax(outputs, dim=2)
    return label_encoder.inverse_transform(predictions.squeeze().numpy())


# Input loop for predictions
while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break

    print(f"Predicted tags: {predict_tags(user_input)}")
