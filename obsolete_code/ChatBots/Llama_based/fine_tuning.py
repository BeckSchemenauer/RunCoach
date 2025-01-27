import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    TrainingArguments,
    Trainer,
    GenerationConfig)
from datasets import load_dataset, DatasetDict, Dataset

# Define model name
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


# Step 1: Load and Tokenize the Dataset
def prepare_dataset(dataset_path):
    # Load dataset from JSON
    raw_dataset = Dataset.from_json(dataset_path)

    # Split dataset into train and test subsets
    dataset_dict = raw_dataset.train_test_split(test_size=0.2, seed=42)

    # Wrap into a DatasetDict
    dataset = DatasetDict({
        "train": dataset_dict["train"],
        "test": dataset_dict["test"]
    })

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Add a padding token to the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})  # Use eos_token as pad_token

    # Tokenization function
    def tokenize_function(example):
        # Tokenize the input-output pairs and create labels
        inputs = tokenizer(
            example['input'],
            text_pair=example['output'],
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # Add labels (shifted input_ids for causal language modeling)
        inputs['labels'] = inputs['input_ids'].copy()  # Copy input_ids to labels
        return inputs

    # Tokenize dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets, tokenizer


# Step 2: Fine-Tune the Model
def fine_tune_model(tokenized_datasets, tokenizer, output_dir="./fine_tuned_model"):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=torch.float16
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        learning_rate=2e-5,  # Learning rate
        per_device_train_batch_size=2,  # Batch size for training
        per_device_eval_batch_size=2,  # Batch size for evaluation
        num_train_epochs=3,  # Number of training epochs
        weight_decay=0.01,  # Weight decay
        logging_dir='./logs',  # Directory for storing logs
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,  # The model being fine-tuned
        args=training_args,  # Training arguments
        train_dataset=tokenized_datasets['train'],  # Training dataset
        eval_dataset=tokenized_datasets['test'],  # Evaluation dataset
        tokenizer=tokenizer,  # The tokenizer used for encoding input/output pairs
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


# Step 3: Chatbot Inference
def chat_with_model(model, tokenizer):
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    print("Chatbot is ready! Type 'exit' to quit.")

    history = ""
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Add user input to the conversation history
        history += f"User: {user_input}\nAssistant: "

        # Tokenize the history and move to device
        inputs = tokenizer(history, return_tensors="pt").to(model.device)

        # Generate a response
        generation_config = GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Decode the output and print the assistant's response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {response}")

        # Add the assistant's response to the conversation history
        history += f"{response}\n"


# Main Function
if __name__ == "__main__":
    # Path to your dataset
    DATASET_PATH = "your_dataset.json"  # Replace with the path to your dataset

    # Step 1: Prepare the dataset
    print("Preparing the dataset...")
    tokenized_datasets, tokenizer = prepare_dataset(DATASET_PATH)

    # Step 2: Fine-tune the model
    print("Fine-tuning the model...")
    model, tokenizer = fine_tune_model(tokenized_datasets, tokenizer)

    # Step 3: Run the chatbot
    print("Starting the chatbot...")
    chat_with_model(model, tokenizer)
