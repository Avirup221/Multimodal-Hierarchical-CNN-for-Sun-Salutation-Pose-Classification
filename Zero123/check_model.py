import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import os

print("--- Generating and Saving Multi-View Grid for Yoga Image ---")

# --- Step 1: Check GPU ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Step 2: Load the Zero123Plus pipeline ---
pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", 
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to(device)

# --- Step 3: Load your input image ---
input_image_path = "/home/avirupd/summer_project/Background_remove/single_test_no_bg.png"

if not os.path.exists(input_image_path):
    raise FileNotFoundError(f"Image not found at: {input_image_path}")

input_image = Image.open(input_image_path).convert("RGB")
print(f"✅ Successfully loaded image: {input_image_path}")

# --- Step 4: Run inference to generate a multi-view grid ---
print("🌀 Generating multi-view grid (this may take a few minutes)...")
result_image = pipeline(input_image, num_inference_steps=28).images[0]

# --- Step 5: Save the output ---
output_path = "multi_view_grid.png"
result_image.save(output_path)

print(f"\n✅ SUCCESS: Multi-view grid image saved to: {os.path.abspath(output_path)}")
