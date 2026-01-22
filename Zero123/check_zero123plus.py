import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import os

print("--- Starting Zero-1-to-3-Plus Model Check ---")

# --- Step 1: Verify GPU ---
print("INFO: Checking for GPU...")
if not torch.cuda.is_available():
    print("FATAL: No GPU detected. This script requires a GPU.")
    exit()
device = "cuda:0"
print(f"SUCCESS: GPU is available. Using device: {device}")

# --- Step 2: Load the Pipeline ---
print("\nINFO: Loading the 'sudo-ai/zero123plus-v1.1' pipeline...")
print("      (This may take a few minutes if downloading for the first time)...")
try:
    pipeline = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.1", 
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    
    # Use the recommended scheduler
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config, timestep_spacing='trailing'
    )
    pipeline.to(device)
    print("SUCCESS: Pipeline loaded successfully.")

except Exception as e:
    print(f"FATAL: Failed to load the pipeline. Error: {e}")
    exit()

# --- Step 3: Get an Input Image ---
print("\nINFO: Downloading a sample input image for the test...")
image_url = "https://d.skis.ltd/nrp/sample-data/lysol.png"
try:
    input_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
    print("SUCCESS: Sample image downloaded.")
except Exception as e:
    print(f"FATAL: Could not download the sample image. Error: {e}")
    exit()

# --- Step 4: Run Inference ---
print("\nINFO: Generating one new view from the sample image...")
try:
    # Running with a low number of steps for a quick test
    result_image = pipeline(input_image, num_inference_steps=28).images[0]
    print("SUCCESS: Inference complete.")
except Exception as e:
    print(f"FATAL: Inference failed. Error: {e}")
    exit()

# --- Step 5: Save the Output ---
output_path = "test_output_zero123plus.png"
result_image.save(output_path)
print(f"\nSUCCESS: The generated image has been saved to '{output_path}'.")

print("\n--- Model Check Complete ---")
print("The Zero-1-to-3-Plus model is working correctly!")

