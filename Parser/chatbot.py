from collections import Counter
import torch
import torch.nn as nn
import nltk
from sklearn.preprocessing import LabelEncoder
import json
from Parser.activity_tagger import activity_tagger
from Parser.classes import User, Activity, Input

nltk.download('punkt')


# Assuming the BiLSTMModel and tokenizer from the original code are present here

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
with open('running_specific/InfoExtractor/activity_details/generated_activity_sentences.json', 'r') as file:
    data = json.load(file)

all_tokens = [token for item in data for token in nltk.word_tokenize(item['sentence'])]
vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}
vocab_size = len(vocab) + 1

label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

embed_dim = 50
hidden_dim = 64
output_dim = len(label_encoder.classes_)
model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)

model.load_state_dict(torch.load('running_specific/InfoExtractor/activity_details/bilstm_model.pth'))
model.eval()


def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


def predict_tags(sentence):
    tokens = [vocab.get(token, 0) for token in tokenizer(sentence)]  # Convert tokens to indices
    tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    outputs = model(tokens_tensor)
    predictions = torch.argmax(outputs, dim=2)  # Get predicted labels for each token
    predicted_labels = label_encoder.inverse_transform(predictions.squeeze().numpy())
    return predicted_labels


# Modify the chatbot to return a dictionary of predicted tags
def chatbot():
    print("Hello! I am your chatbot. Type 'exit' to quit.")

    user = User()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        new_activity = activity_tagger(user_input, predict_tags(user_input))
        user.add_activity(new_activity)
        input_instance = Input(activity=new_activity, intent=None, gear=None, other=None)

        print(input_instance.activity)


# Run the chatbot
chatbot()
