from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import random
import pandas as pd
import nltk
import torch

#nltk.download('punkt')

# Initialize model and tokenizer
model_name = 'facebook/bart-base'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def load_microsoft_data():
    # Load the TSV file
    df = pd.read_csv("ChatBots/BART_based/Release/compressionhistory.tsv", sep="\t")

    # Calculate the sum of AverageGrammar and AverageMeaning
    df["GrammarMeaningSum"] = df["AverageGrammar"] + df["AverageMeaning"]

    # Sort the DataFrame by the sum in descending order
    df_sorted = df.sort_values(by="GrammarMeaningSum", ascending=False)

    # Select the top 512 rows
    top_512 = df_sorted.head(512)

    # Create the new dataset
    data = []

    for _, row in top_512.iterrows():
        data.append({"Text": row["Shortening"], "Label": "short"})
        data.append({"Text": row["Source"], "Label": "long"})

    # Convert the new dataset to a DataFrame
    new_dataset = pd.DataFrame(data)

    # Save to a new file if needed
    #new_dataset.to_csv("processed_dataset.csv", index=False)

    return new_dataset


# Tokenization function
def tokenize_data(examples):
    input_text = ["short sentence: " + rough for rough in examples['short']]
    target_text = examples['long']

    input_encodings = tokenizer(input_text, truncation=True, padding='max_length', max_length=128)
    target_encodings = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)

    target_encodings['labels'] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        for labels in target_encodings['input_ids']
    ]

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['labels']
    }


# Split into train and eval sets
train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2).values()

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_data, batched=True)
eval_dataset = eval_dataset.map(tokenize_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    dataloader_pin_memory=True,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    fp16=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

# Train and save model
trainer.train()
model.save_pretrained('./bart-expander-finetuned')
tokenizer.save_pretrained('./bart-expander-finetuned')
