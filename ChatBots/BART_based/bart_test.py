from transformers import BartTokenizer, BartForConditionalGeneration

# Load the fine-tuned model and tokenizer
model = BartForConditionalGeneration.from_pretrained('bart-finetuned')
tokenizer = BartTokenizer.from_pretrained('bart-finetuned')

def generate_complete_sentence(rough_sentence):
    # Tokenize and generate
    input_ids = tokenizer.encode(rough_sentence, return_tensors="pt", max_length=128, truncation=True)
    generated_ids = model.generate(input_ids, max_length=128, num_beams=4, temperature=0.7)

    # Decode and return the output
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def main():
    print("Enter an incomplete sentence (type 'exit' to quit):")
    while True:
        rough_sentence = input(">> ").strip()
        if rough_sentence.lower() == "exit":
            print("Exiting...")
            break
        if rough_sentence:
            generated_sentence = generate_complete_sentence(rough_sentence)
            print(f"Complete sentence: {generated_sentence}")

if __name__ == "__main__":
    main()
