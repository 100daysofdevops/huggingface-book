from transformers import DistilBertTokenizer, GPT2Tokenizer

# Define the input text
input = "I love using Hugging Face transformers!"

# Download the GPT2 tokenizer
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize the input
gpt_tokens = gpt_tokenizer.tokenize(text=input)

# Repeat for DistilBERT
distil_tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
distil_tokens = distil_tokenizer.tokenize(text=input)

# Compare the output
print(f"GPT tokenizer: {gpt_tokens}")
print(f"DistilBERT tokenizer: {distil_tokens}")
