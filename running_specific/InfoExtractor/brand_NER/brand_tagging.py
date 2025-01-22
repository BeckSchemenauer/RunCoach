import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
import json
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import nltk
from collections import Counter

# Load Data from JSON File
with open('test_sentences.json', 'r') as file:
    data = json.load(file)

# Tokenization function
nltk.download('punkt')
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)

# Collate function for DataLoader
def collate_fn(batch):
    tokens = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    max_len = max(padded_tokens.size(1), padded_labels.size(1))
    padded_tokens = torch.nn.functional.pad(padded_tokens, (0, max_len - padded_tokens.size(1)))
    padded_labels = torch.nn.functional.pad(padded_labels, (0, max_len - padded_labels.size(1)), value=-1)

    return padded_tokens, padded_labels

# Build vocabulary and label encoder
all_tokens = [token for item in data for token in tokenizer(item['sentence'])]
vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}
vocab_size = len(vocab) + 1

label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

# Custom Dataset Class
class SentenceLabelingDataset(Dataset):
    def __init__(self, data, tokenizer, vocab, label_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]['sentence']
        labels = self.data[idx]['labels']
        tokens = self.tokenizer(sentence)
        token_indices = [self.vocab.get(token, 0) for token in tokens]
        label_ids = self.label_encoder.transform(labels)
        return torch.tensor(token_indices), torch.tensor(label_ids)

# Split the data into training and testing datasets
dataset = SentenceLabelingDataset(data, tokenizer, vocab, label_encoder)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

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

# Hyperparameters and Model Initialization
embed_dim = 50
hidden_dim = 64
output_dim = len(label_encoder.classes_)

model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Loop
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for sentences, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(sentences)

        outputs = outputs.view(-1, output_dim)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sentences, labels in test_dataloader:
            outputs = model(sentences)
            outputs = outputs.view(-1, output_dim)
            labels = labels.view(-1)
            test_loss += criterion(outputs, labels).item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Test Loss: {test_loss / len(test_dataloader):.4f}")

# Save the model
torch.save(model.state_dict(), 'bilstm_model.pth')
