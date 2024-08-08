from transformers import pipeline

# Define the pipeline with a QNLI model
classifier = pipeline("text-classification", model="textattack/bert-base-uncased-QNLI")

# Define the premise and question
premise = "Seattle is a city on the West Coast of the United States."
question = "Where is Seattle located?"

# Classify the text
result = classifier(f"{premise} {question}")

# Print the result
print(result)
