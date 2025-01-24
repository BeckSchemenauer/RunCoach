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

# Function to process and get random sentences from a source
def get_random_sentences_from_source(source, is_csv=False, column_name=None, num_sentences=512):
    if is_csv:
        # Load sentences from CSV
        df = pd.read_csv(source)
        sentences = df[column_name].dropna().tolist()
    else:
        # Load sentences from .txt file
        with open(source, "r", encoding="utf-8") as f:
            sentences = nltk.sent_tokenize(f.read())

    # Tokenize sentences and shuffle
    tokenized_sentences = []
    for text in sentences:
        tokenized_sentences.extend(nltk.sent_tokenize(text))

    random.shuffle(tokenized_sentences)
    return tokenized_sentences[:num_sentences]


# Create rough sentences
def create_rough_sentence(sentence, delete_probability=0.2):
    words = sentence.split()
    rough_sentence = [word for word in words if random.random() > delete_probability]
    return ' '.join(rough_sentence)


# Tokenization function
def tokenize_data(examples):
    input_text = ["incomplete sentence: " + rough for rough in examples['rough']]
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


# Get random sentences from TED Talks and new file
processed_sentences_ted = get_random_sentences_from_source("ted-talks/transcripts.csv", is_csv=True,
                                                           column_name="transcript", num_sentences=256)
processed_sentences_hadds = get_random_sentences_from_source("hadds_approach.txt", is_csv=False, num_sentences=512)

# Combine TED Talks and new file sentences
processed_sentences = processed_sentences_ted + processed_sentences_hadds

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
model.save_pretrained('./bart-finetuned')
tokenizer.save_pretrained('./bart-finetuned')
