from transformers import AutoTokenizer

# Define the model to use
model_name = "bert-base-uncased"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example sentence
sentence = "I love using Hugging Face transformers!"

# Tokenize the sentence
tokens = tokenizer(sentence)

# Print the tokens
print(tokens)
