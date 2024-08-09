from transformers import pipeline

# Define the visual Q&A pipeline
visual_qa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")

# Example image path and question
image_path = "panda.JPG"
question = "What type of animal is in this picture?"

# Perform visual question answering
result = visual_qa_pipeline(question=question, image=image_path)

# Print the result
print(result)
