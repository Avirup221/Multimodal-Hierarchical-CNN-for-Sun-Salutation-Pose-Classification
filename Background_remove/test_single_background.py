import torch
from rembg import remove
from PIL import Image
import os

print("--- Starting Single Image Background Removal Test ---")

# --- 1. CONFIGURATION: SET YOUR IMAGE PATHS HERE ---
# IMPORTANT: Change this to the exact image you want to test
INPUT_IMAGE_PATH = "/home/avirupd/summer_project/data_preprocessing/RenamedDataset/valid/video_clip_004/frame_00005.jpg" # <--- ‼ CHANGE THIS

# This will be the clean, transparent-background output file
OUTPUT_IMAGE_PATH = "single_test_no_bg.png" 
# --- End Configuration ---

def main():
    """Main function to run the background removal."""
    
    # --- 2. CHECK FOR GPU ---
    print("INFO: Checking for GPU...")
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected by PyTorch. 'rembg' will run on the CPU, which will be much slower.")
    else:
        print("SUCCESS: GPU is available. 'rembg[gpu]' will be used.")

    # --- 3. LOAD THE IMAGE ---
    print(f"\nINFO: Loading input image from: {INPUT_IMAGE_PATH}")
    try:
        input_image = Image.open(INPUT_IMAGE_PATH)
        print("SUCCESS: Image loaded.")
    except FileNotFoundError:
        print(f"FATAL: Image not found at the specified path. Please check the path in the script.")
        return
    except Exception as e:
        print(f"FATAL: Could not open image. Error: {e}")
        return

    # --- 4. REMOVE BACKGROUND ---
    print("INFO: Removing background...")
    try:
        # The 'remove' function does all the work
        output_image = remove(input_image)
        print("SUCCESS: Background removed.")
    except Exception as e:
        print(f"FATAL: Background removal failed. Error: {e}")
        return

    # --- 5. SAVE THE OUTPUT ---
    output_image.save(OUTPUT_IMAGE_PATH)
    print(f"\n--- Test Complete! ---")
    print(f"SUCCESS: The clean image has been saved to '{OUTPUT_IMAGE_PATH}'")

if __name__ == "__main__":
    main()


### **Step 2: Run the Background Removal Test**

    
