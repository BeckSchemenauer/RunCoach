import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
import json
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification
from transformers import Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence

with open('generated_activity_sentences.json', 'r') as file:
    data = json.load(file)

# Tokenization function using Hugging Face's BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def collate_fn(batch):
    tokens = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Find the maximum sequence length in the batch for padding
    max_len = max([len(t) for t in tokens])

    # Pad tokens and labels to the same length (max_len)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)

    # Adjust the lengths if necessary
    if padded_tokens.size(1) > padded_labels.size(1):
        padded_labels = torch.cat(
            [padded_labels, torch.full((padded_labels.size(0), padded_tokens.size(1) - padded_labels.size(1)), -1)],
            dim=1)
    elif padded_tokens.size(1) < padded_labels.size(1):
        padded_tokens = torch.cat(
            [padded_tokens, torch.full((padded_tokens.size(0), padded_labels.size(1) - padded_tokens.size(1)),
                                       tokenizer.pad_token_id)],
            dim=1)

    return padded_tokens, padded_labels


# Build vocabulary from training data
vocab = tokenizer.get_vocab()
vocab_size = len(vocab)

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

    def __getitem__(self, idx):
        sentence = self.data[idx]['sentence']
        labels = self.data[idx]['labels']

        encoding = self.tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        token_indices = encoding['input_ids'].squeeze(0)

        label_ids = self.label_encoder.transform(labels)
        return torch.tensor(token_indices), torch.tensor(label_ids)


# Split the data into training and testing datasets
dataset = SentenceLabelingDataset(data, tokenizer, label_encoder)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)


# BERT-based Model Definition
class BertForTokenClassificationWithHead(nn.Module):
    def __init__(self, model_name, num_labels):
        super(BertForTokenClassificationWithHead, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    def forward(self, x):
        return self.bert(x).logits


# Hyperparameters and Model Initialization
output_dim = len(label_encoder.classes_)  # Number of unique labels
model = BertForTokenClassificationWithHead('bert-base-uncased', output_dim)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training and Evaluation Loop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

num_epochs = 3
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

        # Ignore padding tokens in loss calculation
        loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels)

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

            loss = nn.CrossEntropyLoss(ignore_index=-1)(outputs, labels)
            test_loss += loss.item()

    # Print losses
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader)}, Test Loss: {test_loss / len(test_dataloader)}")

model.eval()  # Set model to evaluation mode
predictions, true_labels = [], []  # Track predictions and true labels for accuracy calculation

# Map integer labels back to string labels
reverse_label_dict = {v: k for k, v in label_encoder.classes_.items()}  # Map from label index to string label

# Collect some input sentences for displaying predictions
val_sentences = [item['sentence'] for item in data]  # You can adjust this to match the validation set
val_labels = [item['labels'] for item in data]  # Likewise adjust for validation set labels

with torch.no_grad():  # Disable gradient computation for efficiency
    for batch_idx, batch in enumerate(test_dataloader):  # Using test_dataloader for evaluation
        sentences, labels = batch
        sentences = sentences.to(device)  # Move sentences to the correct device
        labels = labels.to(device)  # Move true labels to the correct device

        # Forward pass through BERT model
        outputs = model(sentences)  # Get raw outputs (logits)
        logits = outputs.view(-1, output_dim)  # Flatten logits
        preds = torch.argmax(logits, dim=1)  # Get predicted class indices

        # Convert to numpy and store the predictions and true labels
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.view(-1).cpu().numpy())

        # Display example sentences with predictions and true labels
        for i in range(len(preds)):
            predicted_label = reverse_label_dict[preds[i].item()]  # Convert index to label
            true_label = reverse_label_dict[labels[i].item()]  # Convert index to label
            print(f"Sentence: {val_sentences[batch_idx * len(preds) + i]}")
            print(f"Predicted: {predicted_label}, True: {true_label}")
            print("-" * 50)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)  # Calculate overall accuracy
print(f"Validation Accuracy: {accuracy:.2f}")  # Report accuracy

# Save the model
torch.save(model.state_dict(), 'bert_model.pth')
print("Model saved successfully!")
