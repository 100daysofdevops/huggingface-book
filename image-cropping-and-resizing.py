from PIL import Image

# Load an image
image = Image.open("hf-logo-with-title.png")

# Crop the image (left, upper, right, lower)
cropped_image = image.crop((100, 100, 400, 400))

# Resize the image
resized_image = cropped_image.resize((200, 200))

# Save the processed image
resized_image.save("processed_image.png")
