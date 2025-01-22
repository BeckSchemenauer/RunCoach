import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json


# Dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# Label dictionary to map string labels to integers
label_dict = {"modify": 0, "validate": 1, "generate": 2, "recommend": 3, "train": 4, "track": 5}

# Define the model and tokenizer
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME,
                                                      num_labels=len(label_dict))

# Load the labeled data
file_path = "labeled_questions_dict.json"  # Update the path if needed
with open(file_path, "r") as f:
    data = json.load(f)

# Prepare the examples by iterating over the categories
examples = []
for category, questions in data.items():
    label = label_dict[category]
    for question in questions:
        examples.append({"question": question, "label": label})

# Extract questions and labels
questions = [example["question"] for example in examples]
labels = [example["label"] for example in examples]

# Train-test split with class balance
train_texts, val_texts, train_labels, val_labels = train_test_split(
    questions, labels, test_size=0.2, random_state=42, stratify=labels
)

# Verify the split
print("Training set size:", len(train_texts))
print("Validation set size:", len(val_texts))
print("Class distribution in training set:", {label: train_labels.count(label) for label in set(train_labels)})
print("Class distribution in validation set:", {label: val_labels.count(label) for label in set(val_labels)})

# Create datasets
train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_loss = 0  # Track cumulative loss for the epoch

    for batch in train_loader:
        optimizer.zero_grad()  # Reset gradients
        input_ids = batch["input_ids"].to(device)  # Move input IDs to the correct device
        attention_mask = batch["attention_mask"].to(device)  # Move attention masks to the correct device
        labels = batch["label"].to(device)  # Move labels to the correct device

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)  # Forward pass
        loss = outputs.loss  # Get the loss value
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        total_loss += loss.item()  # Accumulate loss for reporting

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")  # Report average loss for the epoch

# Validation loop
model.eval()  # Set model to evaluation mode
predictions, true_labels = [], []  # Track predictions and true labels for accuracy calculation

# Map integer labels back to string labels
reverse_label_dict = {v: k for k, v in label_dict.items()}

with torch.no_grad():  # Disable gradient computation for efficiency
    for batch_idx, batch in enumerate(val_loader):  # Enumerate to get the index of the batch
        input_ids = batch["input_ids"].to(device)  # Move input IDs to the correct device
        attention_mask = batch["attention_mask"].to(device)  # Move attention masks to the correct device
        labels = batch["label"].to(device)  # Move labels to the correct device

        outputs = model(input_ids, attention_mask=attention_mask)  # Forward pass
        logits = outputs.logits  # Get raw scores (logits)
        preds = torch.argmax(logits, dim=1)  # Convert logits to predicted class indices

        predictions.extend(preds.cpu().numpy())  # Store predictions
        true_labels.extend(labels.cpu().numpy())  # Store true labels

        # Print the input question and the predicted label for each batch
        for i in range(len(preds)):
            predicted_label = reverse_label_dict[preds[i].item()]  # Map predicted integer label back to string
            true_label = reverse_label_dict[labels[i].item()]  # Map true integer label back to string
            print(f"Question: {val_texts[batch_idx * val_loader.batch_size + i]}")
            print(f"Predicted: {predicted_label}, True: {true_label}")
            print("-" * 50)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predictions)  # Calculate accuracy
print(f"Validation Accuracy: {accuracy:.2f}")  # Report validation accuracy


# Save the fine-tuned model
model.save_pretrained("./intent_model")
tokenizer.save_pretrained("./intent_model")
