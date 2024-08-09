from diffusers import StableDiffusionPipeline
import torch

# Load the pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Generate an image from a text prompt
prompt = "A beautiful landscape with mountains and a river"
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_image.png")
image.show()
