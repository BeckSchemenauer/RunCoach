import nltk
from Parser.classes import Activity
from collections import Counter
import torch
import torch.nn as nn
import nltk
from sklearn.preprocessing import LabelEncoder
import json

nltk.download('punkt')


# Define the BiLSTM model
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


# Load vocabulary and label encoder
def load_vocabulary_and_encoder():
    with open('running_specific/InfoExtractor/activity_details/generated_activity_sentences.json', 'r') as file:
        data = json.load(file)

    all_tokens = [token for item in data for token in nltk.word_tokenize(item['sentence'])]
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}
    label_encoder = LabelEncoder()

    all_labels = [label for item in data for label in item["labels"]]
    label_encoder.fit(all_labels)

    return vocab, label_encoder


# Load the model
def load_model(vocab_size, output_dim):
    embed_dim = 50
    hidden_dim = 64
    model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load('running_specific/InfoExtractor/activity_details/bilstm_model.pth'))
    model.eval()
    return model


# Tokenizer function
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


# Predict tags using the trained BiLSTM model
def predict_tags(sentence, model, vocab, label_encoder):
    tokens = [vocab.get(token, 0) for token in tokenizer(sentence)]  # Convert tokens to indices
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    outputs = model(tokens_tensor)
    predictions = torch.argmax(outputs, dim=2)  # Get predicted labels for each token
    predicted_labels = label_encoder.inverse_transform(predictions.squeeze().numpy())
    return predicted_labels


def activity_tagger(sentence, predicted_tags):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)

    # Initialize variables to store the values
    activity = None
    distance = None
    distance_unit = None
    time = None
    pace = None
    pace_unit = None
    elevation = None
    relative_effort = None
    heart_rate = None

    # Loop through predicted tags and assign values to corresponding variables
    for i, tag in enumerate(predicted_tags):
        if tag == 'activity':
            activity = tokens[i]
        elif tag == 'distance':
            distance = tokens[i]
        elif tag == 'unit':
            distance_unit = tokens[i]
        elif tag == 'time':
            time = tokens[i]
        elif tag == 'pace':
            pace = tokens[i]
        elif tag == 'pace-unit':
            pace_unit = tokens[i]

    # Create and return the Activity object with the extracted values
    return Activity(activity, distance, distance_unit, time, pace, pace_unit, elevation, relative_effort, heart_rate)
