import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
import json
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

with open('all_purpose.json', 'r') as file:
    data = json.load(file)

# Tokenization function using Hugging Face's BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def collate_fn(batch):
    tokens = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    attention_masks = [item[2] for item in batch]  # Collect attention masks

    # Find the maximum sequence length in the batch for padding
    max_len = max([len(t) for t in tokens])

    # Pad tokens, labels, and attention masks to the same length (max_len)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)  # Use 0 for padding in attention masks

    if padded_tokens.size(1) > padded_labels.size(1):
        padded_labels = torch.cat(
            [padded_labels, torch.full((padded_labels.size(0), padded_tokens.size(1) - padded_labels.size(1)), -1)],
            dim=1)
        padded_attention_masks = torch.cat(
            [padded_attention_masks, torch.full((padded_attention_masks.size(0), padded_tokens.size(1) - padded_attention_masks.size(1)), 0)],
            dim=1)
    elif padded_tokens.size(1) < padded_labels.size(1):
        padded_tokens = torch.cat(
            [padded_tokens, torch.full((padded_tokens.size(0), padded_labels.size(1) - padded_tokens.size(1)), tokenizer.pad_token_id)],
            dim=1)
        padded_attention_masks = torch.cat(
            [padded_attention_masks, torch.full((padded_attention_masks.size(0), padded_labels.size(1) - padded_attention_masks.size(1)), 0)],
            dim=1)

    return padded_tokens, padded_labels, padded_attention_masks  # Return all three: tokens, labels, attention_masks

# Label Encoder for labels
label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

# Custom Dataset Class
class SentenceLabelingDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    # In the Dataset class or wherever you're processing your data:
    def __getitem__(self, idx):
        tokens = self.data[idx]['tokens']
        labels = self.data[idx]['labels']

        # Print the tokens and their original labels
        print(f"Original Tokens: {tokens}")
        print(f"Original Labels: {labels}")

        # Convert tokens to token indices
        token_indices = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]

        # Encode the labels
        label_ids = self.label_encoder.transform(labels)

        # Print the tokenized indices and their encoded labels
        print(f"Tokenized Indices: {token_indices}")
        print(f"Encoded Labels: {label_ids}")

        attention_mask = [1] * len(token_indices)

        return torch.tensor(token_indices), torch.tensor(label_ids), torch.tensor(attention_mask)


# Split the data into training and testing datasets
dataset = SentenceLabelingDataset(data, tokenizer, label_encoder)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

# BERT-based Model Definition (using Hugging Face's pre-trained model directly)
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and Evaluation Loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

num_epochs = 8
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    test_loss = 0

    # Training loop
    for batch in train_dataloader:
        sentences, labels, attention_mask = batch
        sentences = sentences.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)  # Move attention mask to the correct device

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sentences, attention_mask=attention_mask)  # Pass attention_mask as well

        # Reshape outputs and labels
        outputs = outputs.logits.view(-1, len(label_encoder.classes_))
        labels = labels.view(-1)

        # Ignore padding tokens in loss calculation
        loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Evaluation loop
    with torch.no_grad():
        for batch in test_dataloader:
            sentences, labels, attention_mask = batch  # Add attention_mask here
            sentences = sentences.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)  # Move attention mask to the correct device

            outputs = model(sentences, attention_mask=attention_mask)  # Pass attention_mask as well

            # Calculate loss
            outputs = outputs.logits.view(-1, len(label_encoder.classes_))
            labels = labels.view(-1)

            loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels)
            test_loss += loss.item()

    # Print losses
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")

# Save the fine-tuned model
model.save_pretrained("./tagger_model")
tokenizer.save_pretrained("./tagger_model")

print("Model saved successfully!")
