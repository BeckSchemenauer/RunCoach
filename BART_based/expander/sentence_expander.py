from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import torch
import torch.nn.functional as F

# Initialize model and tokenizer
model_name = 'facebook/bart-base'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)


def load_microsoft_data():
    # Load the TSV file and skip malformed rows
    df = pd.read_csv("Release/compressionhistory.tsv", sep="\t", on_bad_lines='skip')

    # Drop rows where 'Shortening' is an integer
    df = df[df["Shortening"].apply(lambda x: not isinstance(x, (int, float)))]

    # Calculate the sum of AverageGrammar and AverageMeaning
    df["GrammarMeaningSum"] = df["AverageGrammar"] + df["AverageMeaning"]

    # Sort the DataFrame by the sum in descending order
    df_sorted = df.sort_values(by="GrammarMeaningSum", ascending=False)

    # Select the top 512 rows
    top_512 = df_sorted.head(512)

    # Create a list of dictionaries for the dataset
    data = [
        {"short": row["Shortening"], "long": row["Source"]}
        for _, row in top_512.iterrows()
    ]

    # Convert the data into a Hugging Face Dataset
    return Dataset.from_list(data)


# Load the dataset
dataset = load_microsoft_data()

# Split into train and eval sets
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']


# Tokenization function
def tokenize_data(examples):
    input_text = ["short sentence: " + s for s in examples['short']]
    target_text = examples['long']

    input_encodings = tokenizer(input_text, truncation=True, padding='max_length', max_length=128)
    target_encodings = tokenizer(target_text, truncation=True, padding='max_length', max_length=128)

    # Replace padding token ID with -100 for labels
    target_encodings['labels'] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels]
        for labels in target_encodings['input_ids']
    ]

    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['labels']
    }


# Tokenize datasets
train_dataset = train_dataset.map(tokenize_data, batched=True, remove_columns=['short', 'long'])
eval_dataset = eval_dataset.map(tokenize_data, batched=True, remove_columns=['short', 'long'])


# Custom Trainer class to reward longer outputs
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute the CrossEntropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, model.config.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        # Reward longer outputs
        predicted_lengths = (logits.argmax(dim=-1) != tokenizer.pad_token_id).sum(dim=1)
        target_lengths = (labels != -100).sum(dim=1)
        length_penalty = torch.abs(predicted_lengths - target_lengths).float().mean()

        # Combine the two losses
        loss = ce_loss + 0.1 * length_penalty  # Adjust the weight of the length penalty as needed

        return (loss, outputs) if return_outputs else loss



# Training arguments
training_args = TrainingArguments(
    output_dir='../sentence_polisher/results',
    num_train_epochs=5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    fp16=True,
    dataloader_pin_memory=True
)

# Initialize Custom Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# Train and save model
trainer.train()
model.save_pretrained('./bart-expander-finetuned')
tokenizer.save_pretrained('./bart-expander-finetuned')
