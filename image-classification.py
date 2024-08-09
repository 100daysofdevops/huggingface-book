from transformers import pipeline
from PIL import Image

# Define the image classification pipeline
classifier = pipeline("image-classification")

# Load an image (ensure the image path is correct)
image_path = "processed_image.png"

# Load the image using PIL
image = Image.open(image_path)

# Classify the image, passing the PIL Image object
result = classifier(image)  # Pass the image object directly

# Print the classification result
print(result)
