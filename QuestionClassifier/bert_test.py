import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load the fine-tuned model and tokenizer
MODEL_PATH = "./intent_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Set the model to evaluation mode
model.eval()

# Define the label dictionary to map numeric class indices back to text labels
label_dict = {0: "modify", 1: "validate", 2: "generate", 3: "recommend", 4: "train", 5: "track"}


# Function to predict the class of the input text
def predict_class(input_text):
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


# Main function to take input and predict class in a loop
if __name__ == "__main__":
    while True:
        # Ask the user for input
        user_input = input("Enter a question (or type 'exit' to quit): ")

        # Check if the user wants to exit the loop
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        # Predict and display the result
        predicted_label = predict_class(user_input)
        print(f"The predicted class is: {predicted_label}")
