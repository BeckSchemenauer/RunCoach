import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Define the label dictionary to map numeric class indices back to text labels
label_dict = {0: "modify", 1: "validate", 2: "generate", 3: "recommend", 4: "train", 5: "track"}


def load_intent_model(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return tokenizer, model


def classify_intent(input_text, tokenizer, model):
    # Tokenize the input text
    encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Make prediction
    with torch.no_grad():
        output = model(**encoding)
        logits = output.logits  # Get logits (raw predictions)

        # Convert logits to probabilities (softmax)
        probs = torch.nn.functional.softmax(logits, dim=1)

        # Get the predicted class (index of highest probability)
        predicted_class_idx = torch.argmax(probs, dim=1).item()

        # Ensure the predicted class index is valid
        if predicted_class_idx in label_dict:
            predicted_label = label_dict[predicted_class_idx]
        else:
            predicted_label = "Unknown"  # Default case if the index is unexpected

        return predicted_label
