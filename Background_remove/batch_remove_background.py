import torch
from rembg import remove
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
# Path to your master dataset
RENAMED_DATASET_PATH = "/home/avirupd/summer_project/data_preprocessing/RenamedDataset"

# Path to your master label CSVs
LABEL_CSVS_PATH = "/home/avirupd/summer_project"

# The new directory we will create to store the clean images
OUTPUT_PATH = "ProcessedData/1_RembgOutput"

# The specific video clips to process
TARGET_CLIPS = {"video_clip_001", "video_clip_002", "video_clip_005"}
# ---------------------

def load_master_label_map(csvs_path):
    """Loads all label CSVs into one master dictionary."""
    print(f"Loading master label maps from: {csvs_path}")
    master_map = {}
    csv_files = ["labeled_data.csv", "labeled_data_test.csv", "labeled_data_valid.csv"]
    
    for f in csv_files:
        try:
            df = pd.read_csv(os.path.join(csvs_path, f))
            # Create a dictionary of {original_filename: label}
            for _, row in df.iterrows():
                master_map[row['filename']] = row['label']
        except FileNotFoundError:
            print(f"Warning: Could not find label file {f}")
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not master_map:
        raise ValueError("No label data was loaded. Please check CSV paths.")
    
    print(f"Loaded {len(master_map)} labels into master map.")
    return master_map

def process_pipeline():
    # 1. Load the master key for all labels
    label_map = load_master_label_map(LABEL_CSVS_PATH)
    
    # 2. Loop through train/test/valid splits
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(RENAMED_DATASET_PATH, split)
        if not os.path.exists(split_path):
            print(f"Skipping split '{split}': directory not found.")
            continue
            
        print(f"\nProcessing split: {split}")
        
        # 3. Loop through each video clip folder in the split
        for clip_name in os.listdir(split_path):
            clip_path = os.path.join(split_path, clip_name)
            
            # 4. Check if this is one of the clips we want to process
            if clip_name not in TARGET_CLIPS:
                print(f"  - Skipping {clip_name} (not in target list)")
                continue
            
            print(f"  + Processing target clip: {clip_name}")
            
            # 5. Load the *local* frame map for this clip
            try:
                map_file = os.path.join(clip_path, f"{clip_name}_frame_map.csv")
                frame_map_df = pd.read_csv(map_file)
                # Create a dict: {new_filename: original_filename}
                local_map = dict(zip(frame_map_df['new_filename'], frame_map_df['original_filename']))
            except Exception as e:
                print(f"    ERROR: Could not load frame map for {clip_name}. Skipping. Error: {e}")
                continue

            # 6. Process each frame in the clip folder
            frame_files = [f for f in os.listdir(clip_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for new_filename in tqdm(frame_files, desc=f"  {clip_name}", leave=False):
                try:
                    # --- Label Lookup ---
                    # a. Find original_filename using the local map
                    original_filename = local_map.get(new_filename)
                    if not original_filename:
                        # print(f"Warning: No map found for {new_filename}. Skipping.")
                        continue
                        
                    # b. Find the label using the master map
                    label = label_map.get(original_filename)
                    if not label:
                        # print(f"Warning: No master label found for {original_filename}. Skipping.")
                        continue
                    
                    # --- Background Removal ---
                    # c. Create the final output path, sorted by label
                    output_label_dir = os.path.join(OUTPUT_PATH, split, label)
                    os.makedirs(output_label_dir, exist_ok=True)
                    
                    # We save as PNG to preserve transparency
                    output_image_path = os.path.join(output_label_dir, f"{clip_name}_{new_filename}.png")
                    
                    # d. Skip if we've already done this file
                    if os.path.exists(output_image_path):
                        continue
                        
                    # e. Load the original image
                    input_image_path = os.path.join(clip_path, new_filename)
                    input_image = Image.open(input_image_path)
                    
                    # f. Run rembg
                    output_image = remove(input_image)
                    
                    # g. Save the clean PNG
                    output_image.save(output_image_path)
                    
                except Exception as e:
                    print(f"Error processing {new_filename}: {e}")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected. 'rembg' will run on CPU (this will be very slow).")
    else:
        print("INFO: GPU detected. 'rembg[gpu]' will be used.")
        
    process_pipeline()
    print("\n--- Background removal batch job complete! ---")
    