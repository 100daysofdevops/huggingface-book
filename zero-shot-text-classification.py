from transformers import pipeline

# Define the pipeline
classifier = pipeline("zero-shot-classification")

# Define the text and candidate labels
text = "The recent advances in AI are groundbreaking."
candidate_labels = ["politics", "science", "technology"]

# Classify the text
result = classifier(text, candidate_labels)

# Print the result
print(result)
