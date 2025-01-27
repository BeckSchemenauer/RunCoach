import torch
from transformers import BertTokenizer, BertForTokenClassification
import json
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('punkt')  # Ensure that necessary NLTK resources are downloaded
from nltk.tokenize import word_tokenize

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./tagger_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForTokenClassification.from_pretrained(MODEL_PATH)

# Load the activity sentences data
with open('all_purpose.json', 'r') as file:
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


def split_colon_tokens(sentence):
    # Split tokens containing a colon (e.g., "5:00" -> ["5", ":", "00"])
    words = sentence.split()
    split_words = []

    for word in words:
        if ':' in word:
            # Split by colon and keep the parts
            parts = word.split(':')
            split_words.extend(parts)  # Add the parts separately
        else:
            split_words.append(word)  # Keep the word as is
    return ' '.join(split_words)


def predict_tags(sentence):
    # First, split tokens containing a colon
    sentence = split_colon_tokens(sentence)

    # Tokenize the sentence using NLTK
    nltk_tokens = word_tokenize(sentence)
    print(f"NLTK Tokens: {nltk_tokens}")

    # Convert NLTK tokens into token IDs using the BERT tokenizer
    encoding = tokenizer(nltk_tokens, padding='max_length', truncation=True, max_length=128, is_split_into_words=True,
                         return_tensors='pt')

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
    print("\nPredicted tags for sentence: ")
    for token, label in zip(decoded_tokens, predicted_labels):
        print(f"{token}: {label}")


# Terminal I/O loop to input multiple sentences
while True:
    sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
    if sentence.lower() == 'exit':
        print("Exiting...")
        break
    else:
        predict_tags(sentence)
