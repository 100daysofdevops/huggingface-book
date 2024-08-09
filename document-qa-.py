from transformers import pipeline

# Define the document Q&A pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Example document and question
document = """
In this research paper, we discuss the various impacts of climate change on agriculture.
Key action steps include adopting sustainable farming practices, reducing greenhouse gas emissions,
and increasing research on climate-resistant crops.
"""

question = "What are the action steps?"

# Perform question answering
result = qa_pipeline(question=question, context=document)

# Print the result
print(result)
