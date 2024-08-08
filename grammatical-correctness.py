from transformers import pipeline

# Define the pipeline with a grammatical correctness model
classifier = pipeline("text-classification", model="textattack/bert-base-uncased-CoLA")

# Classify the text
result = classifier("Enjoys she books reading")

# Print the result
print(result)
