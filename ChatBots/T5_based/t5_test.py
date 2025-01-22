from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('./t5-finetuned')
tokenizer = T5Tokenizer.from_pretrained('./t5-finetuned')

# Example rough sentence
rough_sentence = "Machine learning applications grow everyday."

# Tokenize and generate
input_ids = tokenizer.encode(rough_sentence, return_tensors="pt", max_length=128, truncation=True)
generated_ids = model.generate(input_ids, max_length=128, num_beams=4, temperature=0.7)

# Decode and print the output
generated_sentence = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(generated_sentence)