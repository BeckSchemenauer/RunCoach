import torch
from transformers import BertTokenizer, BertForTokenClassification
import json
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
from Parser.classes import Activity


def load_all_purpose_model(model_path='running_specific/InfoExtractor/all_purpose/tagger_model', json_path='running_specific/InfoExtractor/all_purpose/all_purpose.json'):
    """Load the model, tokenizer, and label encoder."""
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForTokenClassification.from_pretrained(model_path)

    # Load data and fit label encoder
    with open(json_path, 'r') as file:
        data = json.load(file)

    label_encoder = LabelEncoder()
    all_labels = [label for item in data for label in item["labels"]]
    label_encoder.fit(all_labels)

    # Set up the model on the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, label_encoder, device


def custom_tokenize(sentence):
    """Splits tokens containing colons into separate parts, keeping the colon as a separate token."""
    split_words = []
    for word in sentence.split():
        if ':' in word:
            parts = word.split(':')
            split_words.extend([part + (':' if i < len(parts) - 1 else '') for i, part in enumerate(parts)])
        else:
            split_words.append(word)
    return word_tokenize(' '.join(split_words))


def predict_tags(sentence, tokenizer, model, label_encoder, device):
    """Predicts tags for a given sentence."""
    # Preprocess and tokenize the sentence
    nltk_tokens = custom_tokenize(sentence)

    # Encode tokens using the BERT tokenizer
    encoding = tokenizer(nltk_tokens, padding='max_length', truncation=True, max_length=128,
                         is_split_into_words=True, return_tensors='pt')

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Run the model
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    # Get predicted class indices and decode labels
    predictions = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
    predicted_labels = label_encoder.inverse_transform(predictions)

    # Remove special tokens and align predictions with tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())
    tokens = [token for token in tokens if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]]

    # Adjust labels: Shift predicted labels by one earlier
    predicted_labels = predicted_labels[:len(tokens)]
    if len(predicted_labels) > 1:
        predicted_labels = predicted_labels[1:]

    # Print results
    print("\nPredicted tags:")
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")

    return predicted_labels


def run_inference():
    tokenizer, model, label_encoder, device = load_all_purpose_model()

    while True:
        sentence = input("\nEnter a sentence (or type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            print("Exiting...")
            break
        predict_tags(sentence, tokenizer, model, label_encoder, device)


def activity_tagger(sentence, predicted_tags):
    # Tokenize the sentence
    tokens = custom_tokenize(sentence)

    # Initialize a dictionary to store the values
    attributes = {
        'activity': None,
        'distance': None,
        'unit': None,
        'time': None,
        'pace': None,
        'pace_unit': None,
        'elevation': None,
        'relative_effort': None,
        'heart_rate': None
    }

    # Loop through predicted tags and assign values to corresponding variables
    for i, tag in enumerate(predicted_tags):
        if tag in attributes:
            attributes[tag] = tokens[i]

    # Return None if all attributes remain None
    if all(value is None for value in attributes.values()):
        return None

    # Create and return the Activity object with the extracted values
    return Activity(**attributes)

