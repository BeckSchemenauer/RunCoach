from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32,
)

# Device for model execution
device = next(model.parameters()).device

# Print the device being used by the model
print(f"Model is loaded on: {device}")

# Global context for every query
GLOBAL_CONTEXT = (
    "Make the sentence a complete sentence and make it more polite."
)


# Prepare LLaMA input
def prepare_llama_input(user_input):
    """
    Prepares the LLaMA input string from the standardized question

    Args:
        user_input (str): The standardized question for LLaMA.
    Returns:
        str: The input string for LLaMA.
    """

    llama_input = (
        f"{GLOBAL_CONTEXT}: {user_input}\n"
    )
    return llama_input


# Dynamic Parameter Function
def get_generation_config(user_input):
    """
    Determines generation parameters dynamically based on the user's input.

    Args:
        user_input (str): The user's latest question or statement.

    Returns:
        dict: A dictionary containing dynamic generation parameters.
    """
    # Default parameters
    config = {
        "max_new_tokens": 70,
        "temperature": 0.9,
        #"top_p": 0.3,
        #"top_k": 30,
        "repetition_penalty": 1.15,
    }

    return config


# Chatbot Function
def chat_with_llama():
    print("LLaMA 3.2 Chatbot initialized! Type 'exit' to end the chat.")

    while True:
        # Step 1: Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break


        # Step 3: Prepare LLaMA input
        llama_input = prepare_llama_input(user_input)

        # Step 4: Tokenize input for LLaMA
        inputs = tokenizer(llama_input, return_tensors="pt").to(device)

        # Step 5: Determine dynamic generation parameters
        dynamic_config = get_generation_config(user_input)
        generation_config = GenerationConfig(
            max_new_tokens=dynamic_config["max_new_tokens"],
            temperature=dynamic_config["temperature"],
            #top_p=dynamic_config["top_p"],
            #top_k=dynamic_config["top_k"],
            repetition_penalty=dynamic_config["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  # Ensure it ends at EOS token
            early_stopping=True  # Stops generation once EOS token is found
        )

        # Step 6: Generate response
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=generation_config)

        # Step 7: Decode and display the response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f"Assistant: {response}")


# Run the chatbot
if __name__ == "__main__":
    chat_with_llama()
