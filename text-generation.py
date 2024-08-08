# Use a pipeline as a high-level helper
from transformers import pipeline

# Initialize the text generation pipeline with the GPT-2 model
pipe = pipeline("text-generation", model="openai-community/gpt2")

# Generate text based on a prompt
output = pipe("AI is going to change the")

# Print the generated text
print(output)
