import torch
from transformers import GPT2Tokenizer, AutoModelForCausalLM

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define the input text
input_text = "AI is going to change the"

# Encode the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate the next word
with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits

# Get the predicted next token
next_token_logits = logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1).item()
predicted_word = tokenizer.decode(next_token_id)

# Print the result
print(f"Input: {input_text}")
print(f"Predicted next word: {predicted_word}")
