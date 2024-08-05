from transformers import GPT2Tokenizer

# Load the GPT2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Example sentence
sentence = "I love using Hugging Face transformers!"

# Tokenize the sentence
tokens = tokenizer(sentence)

# Print the tokens
print(tokens)
