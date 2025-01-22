from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import random
import pandas as pd
import nltk
import torch

print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

nltk.download('punkt_tab')

# Initialize model and tokenizer
model_name = 't5-base'
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load dataset
df = pd.read_csv("ted-talks/transcripts.csv")
sentences = df['transcript'].dropna().tolist()

# Tokenize sentences into individual ones
processed_sentences = []
for talk in sentences:
    processed_sentences.extend(nltk.sent_tokenize(talk))

# Limit to 1000 sentences
processed_sentences = processed_sentences[:512]

# Create rough sentences
def create_rough_sentence(sentence, delete_probability=0.3):
    words = sentence.split()
    rough_sentence = [word for word in words if random.random() > delete_probability]
    return ' '.join(rough_sentence)


# Tokenization function
def tokenize_data(examples):
    input_text = ["correct English sentence: " + rough for rough in examples['rough']]
    target_text = examples['complete']

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

# Generate dataset
rough_sentences = [create_rough_sentence(sentence) for sentence in processed_sentences]
train_data = [{'rough': rough, 'complete': complete} for rough, complete in zip(rough_sentences, processed_sentences)]
train_dataset = Dataset.from_list(train_data)

# Split into train and eval sets
train_dataset, eval_dataset = train_dataset.train_test_split(test_size=0.2).values()

# Tokenize datasets
train_dataset = train_dataset.map(tokenize_data, batched=True)
eval_dataset = eval_dataset.map(tokenize_data, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    dataloader_pin_memory=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    fp16=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

print(model.device)

# Train and save model
trainer.train()
model.save_pretrained('./t5-finetuned')
tokenizer.save_pretrained('./t5-finetuned')
