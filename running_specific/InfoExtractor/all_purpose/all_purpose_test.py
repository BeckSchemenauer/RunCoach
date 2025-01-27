import torch
from transformers import BertTokenizer, BertForTokenClassification
import json
from sklearn.preprocessing import LabelEncoder

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./tagger_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)


with open('generated_activity_sentences.json', 'r') as file:
    data = json.load(file)

# Label Encoder for labels
label_encoder = LabelEncoder()
all_labels = [label for item in data for label in item["labels"]]
label_encoder.fit(all_labels)

# Set the model to evaluation mode
model.eval()

# Move model to the correct device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def predict_tags(sentence):
    # Tokenize the sentence
    encoding = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    # Get input IDs and attention mask
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the predicted token class indices (logits to labels)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze(0)

    # Decode the token labels
    predicted_labels = label_encoder.inverse_transform(predictions.cpu().numpy())

    # Decode tokens back to words (remove padding and special tokens)
    decoded_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    decoded_tokens = [token for token in decoded_tokens if
                      token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]]

    # Print the sentence with the predicted labels
    for token, label in zip(decoded_tokens, predicted_labels):
        print(f"{token}: {label}")


# Example sentence for prediction
sentence = "I ran for 30 minutes and covered 8 kilometers"
predict_tags(sentence)
