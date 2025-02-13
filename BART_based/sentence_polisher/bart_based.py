import json
import random
import pandas as pd
import nltk
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, TrainingArguments, Trainer
from datasets import Dataset

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

# Function to create rough sentences
def create_rough_sentence(sentence, delete_probability=0.3, adjective_boost=0.4):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)  # Get parts of speech for each word

    rough_sentence = []
    for word, pos in pos_tags:
        if pos.startswith('JJ'):  # 'JJ', 'JJR', 'JJS' are tags for adjectives
            if random.random() > delete_probability + adjective_boost:
                rough_sentence.append(word)
        else:
            if random.random() > delete_probability:
                rough_sentence.append(word)

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

# Load JSON data
with open("manual_data.json", "r", encoding="utf-8") as f:
    workout_data = json.load(f)

# Combine JSON entries into one list of sentences
json_sentences = []
for key, values in workout_data.items():
    json_sentences.extend(values)

# Get random sentences from TED Talks and additional file
processed_sentences_ted = get_random_sentences_from_source("ted-talks/transcripts.csv", is_csv=True,
                                                           column_name="transcript", num_sentences=512)
processed_sentences_hadds = get_random_sentences_from_source("hadds_approach.txt", is_csv=False, num_sentences=512)

# Combine sentences from all sources
processed_sentences = processed_sentences_ted + processed_sentences_hadds + json_sentences

# Generate rough sentences and dataset
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
    output_dir="results",
    dataloader_pin_memory=True,
    num_train_epochs=4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.005,
    evaluation_strategy="epoch",
    learning_rate=1e-3,
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
