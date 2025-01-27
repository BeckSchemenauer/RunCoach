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

# Step 1: Load Data from JSON File

# Load the JSON file (you should replace the path with the actual path to your file)
with open('generated_activity_sentences.json', 'r') as file:
    data = json.load(file)


# Tokenization function using NLTK
def tokenizer(sentence):
    return nltk.word_tokenize(sentence)


def collate_fn(batch):
    tokens = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find the maximum sequence length in the batch for padding
    max_len = max([len(t) for t in tokens])

    # Pad tokens and labels to the same length (max_len)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    # If the lengths are different after padding, adjust
    if padded_tokens.size(1) > padded_labels.size(1):
        padded_labels = torch.cat(
            [padded_labels, torch.full((padded_labels.size(0), padded_tokens.size(1) - padded_labels.size(1)), -1)],
            dim=1)
    elif padded_tokens.size(1) < padded_labels.size(1):
        padded_tokens = torch.cat(
            [padded_tokens, torch.full((padded_tokens.size(0), padded_labels.size(1) - padded_tokens.size(1)), 0)],
            dim=1)

    return padded_tokens, padded_labels


# Build vocabulary from training data
all_tokens = [token for item in data for token in tokenizer(item['sentence'])]
vocab = {word: idx + 1 for idx, (word, _) in enumerate(Counter(all_tokens).items())}  # Reserve 0 for padding
vocab_size = len(vocab) + 1  # Adding 1 for padding

# Label Encoder for labels (you can adjust to match the label set)
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
        token_indices = [self.vocab.get(token, 0) for token in tokens]  # Map tokens to indices
        label_ids = self.label_encoder.transform(labels)
        return torch.tensor(token_indices), torch.tensor(label_ids)


# Step 2: Split the data into training and testing datasets
dataset = SentenceLabelingDataset(data, tokenizer, vocab, label_encoder)

# Split into training (80%) and testing (20%) datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Prepare DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)


# Step 3: BiLSTM Model Definition
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # Padding idx is 0
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)  # Output layer (2*hidden_dim due to BiLSTM)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out)
        return out


# Step 4: Hyperparameters and Model Initialization
embed_dim = 50
hidden_dim = 64
output_dim = len(label_encoder.classes_)  # Number of unique labels

model = BiLSTMModel(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding tokens in the labels
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: Training and Evaluation Loop
num_epochs = 15

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    test_loss = 0

    # Training phase
    for batch in train_dataloader:
        sentences, labels = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(sentences)

        # Reshape outputs and labels
        outputs = outputs.view(-1, output_dim)
        labels = labels.view(-1)

        # Debugging shape mismatch
        if outputs.shape[0] != labels.shape[0]:
            print(f"Shape mismatch: Outputs {outputs.shape}, Labels {labels.shape}")
            continue  # Skip this batch to avoid crashing

        print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Evaluation phase
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            sentences, labels = batch
            outputs = model(sentences)

            # Calculate loss
            outputs = outputs.view(-1, output_dim)
            labels = labels.view(-1)

            print(f"eval: Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")

            loss = criterion(outputs, labels)
            test_loss += loss.item()

    # Print losses
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")

# Save the model
torch.save(model.state_dict(), 'bilstm_model.pth')
print("Model saved successfully!")
