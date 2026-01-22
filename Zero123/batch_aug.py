import torch
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
import os
from tqdm import tqdm
import argparse

# --- Configuration ---
# This script reads from the output of Script 1
INPUT_PATH = "/home/avirupd/summer_project/Background_remove/ProcessedData/1_RembgOutput"

# This is the final, augmented dataset
OUTPUT_PATH = "ProcessedData/FINAL_AUGMENTED_DATASET"

# Slicing grid dimensions (based on our 'multi_view_grid_2.jpg' test)
NUM_ROWS = 3
NUM_COLS = 2
# ---------------------

def slice_grid_in_memory(grid_image):
    """
    Slices a 3x2 grid image in memory and returns a list of 6 PIL images.
    """
    slices = []
    total_width, total_height = grid_image.size
    
    # Calculate the size of each individual slice
    slice_width = total_width // NUM_COLS
    slice_height = total_height // NUM_ROWS

    # Loop through the rows, then columns, to match the grid layout
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            # Calculate the bounding box for the slice
            left = c * slice_width
            top = r * slice_height
            right = (c + 1) * slice_width
            bottom = (r + 1) * slice_height
            
            box = (left, top, right, bottom)
            
            # Crop the image
            single_view = grid_image.crop(box)
            slices.append(single_view)
    return slices

def process_augmentation(num_steps):
    """
    Loads the pipeline and processes all images from the INPUT_PATH.
    """
    # 1. Setup the GPU and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: This script requires a GPU. Running on CPU will be extremely slow.")
        
    print("INFO: Loading the 'sudo-ai/zero123plus-v1.1' pipeline...")
    print("      (This may take a few minutes if downloading)...")
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.1", 
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16
        )
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing='trailing'
        )
        pipeline.to(device)
        print("SUCCESS: Pipeline loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load pipeline. Error: {e}")
        print("Please make sure you are in the correct Conda environment (gen_ai_env).")
        return

    # 2. Walk through the clean, labeled folders from Script 1
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(INPUT_PATH, split)
        if not os.path.exists(split_path):
            print(f"Skipping split '{split}': directory not found.")
            continue
            
        print(f"\nProcessing split: {split}")
        
        # 3. Loop through each label folder
        label_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        if not label_folders:
            print(f"  No label folders found in {split_path}. Skipping.")
            continue
            
        for label_name in label_folders:
            label_path = os.path.join(split_path, label_name)
                
            print(f"  + Processing label: {label_name}")
            
            # Create the final output directory for this split/label
            output_label_dir = os.path.join(OUTPUT_PATH, split, label_name)
            os.makedirs(output_label_dir, exist_ok=True)
            
            # 4. Loop through each clean .png file
            clean_images = [f for f in os.listdir(label_path) if f.endswith(".png")]
            if not clean_images:
                print(f"    No clean images found in {label_path}. Skipping.")
                continue
            
            for clean_filename in tqdm(clean_images, desc=f"    {label_name}", leave=False):
                try:
                    # The original filename, e.g., "video_clip_001_frame_00001.png"
                    # We strip the .png to get the base name
                    base_name = os.path.splitext(clean_filename)[0]
                    
                    # --- RESUME CAPABILITY ---
                    # Check if the *first* view for this image already exists. If so, skip.
                    test_output_path = os.path.join(output_label_dir, f"{base_name}_view_01.png")
                    if os.path.exists(test_output_path):
                        continue
                        
                    # --- Run the Pipeline ---
                    input_image_path = os.path.join(label_path, clean_filename)
                    input_image = Image.open(input_image_path)
                    
                    # a. Generate the grid image in memory
                    grid_image = pipeline(input_image, num_inference_steps=num_steps).images[0]
                    
                    # b. Slice the grid in memory
                    sliced_images = slice_grid_in_memory(grid_image) # Returns list of 6 PIL images
                    
                    # c. Save all 6 new images
                    for i, slice_img in enumerate(sliced_images):
                        output_filename = f"{base_name}_view_{i+1:02d}.png"
                        output_save_path = os.path.join(output_label_dir, output_filename)
                        slice_img.save(output_save_path)
                        
                except Exception as e:
                    print(f"\n    ERROR processing {clean_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch augment and slice images.")
    parser.add_argument(
        "--steps", 
        type=int, 
        default=75, 
        help="Number of inference steps. 75 gives good quality, 28 is faster."
    )
    args = parser.parse_args()
    
    process_augmentation(num_steps=args.steps)
    
    print("\n--- Augmentation and Slicing batch job complete! ---")
    print(f"Your final augmented dataset is ready in: {os.path.abspath(OUTPUT_PATH)}")