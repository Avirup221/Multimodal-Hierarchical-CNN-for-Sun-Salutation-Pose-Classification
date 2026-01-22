from PIL import Image
import os

print("--- Starting Grid Slicer ---")

# --- 1. CONFIGURATION ---
# The big grid image you just generated
INPUT_IMAGE_FILE = "multi_view_grid.png" 

# The folder where the 6 separate images will be saved
OUTPUT_FOLDER = "final_sliced_images"

# --- Grid Dimensions (based on your 'multi_view_grid_2.jpg' example) ---
NUM_ROWS = 3
NUM_COLS = 2
# -------------------------

def slice_image_grid():
    try:
        # Load the main grid image
        grid_image = Image.open(INPUT_IMAGE_FILE)
        print(f"✅ Successfully loaded grid image: {INPUT_IMAGE_FILE}")
    except FileNotFoundError:
        print(f"FATAL: Could not find the input file: {INPUT_IMAGE_FILE}")
        print("Please make sure it's in the same directory as this script.")
        return
    except Exception as e:
        print(f"FATAL: Could not open image. Error: {e}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get the total size of the grid image
    total_width, total_height = grid_image.size
    
    # Calculate the size of each individual slice
    slice_width = total_width // NUM_COLS
    slice_height = total_height // NUM_ROWS

    print(f"Grid size: {NUM_ROWS} rows x {NUM_COLS} cols")
    print(f"Calculated slice size: {slice_width}W x {slice_height}H")

    image_counter = 1
    
    # Loop through the rows
    for r in range(NUM_ROWS):
        # Loop through the columns
        for c in range(NUM_COLS):
            
            # Calculate the bounding box for the slice
            left = c * slice_width
            top = r * slice_height
            right = (c + 1) * slice_width
            bottom = (r + 1) * slice_height
            
            box = (left, top, right, bottom)
            
            # Crop the image
            single_view = grid_image.crop(box)
            
            # Save the individual slice
            output_filename = f"view_{image_counter:02d}.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            single_view.save(output_path)
            
            print(f"  -> Saved {output_path}")
            image_counter += 1

    print(f"\n✅ SUCCESS: Sliced {INPUT_IMAGE_FILE} into {image_counter-1} separate images.")
    print(f"Check the '{OUTPUT_FOLDER}' directory.")

if __name__ == "__main__":
    slice_image_grid()


### **How to Run It**
"""
This script only needs the `Pillow` library, which is already installed in all of our environments. You can run this in **any** of your working environments (`gen_ai_env`, `zero123_final_env`, or `py311env`).

1.  **Make sure your environment is active** (e.g., `conda activate gen_ai_env`).
2.  **Make sure the `multi_view_grid_CLEAN.png` file is in the same directory.**
3.  **Run the script:**
    ```bash
    python slice_grid.py

    """