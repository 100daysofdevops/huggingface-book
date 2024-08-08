from transformers import pipeline

# Define the pipeline
classifier = pipeline("text-classification")

# Classify the text
result = classifier("I love it")

# Print the result
print(result)
