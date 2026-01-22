import os
import shutil
import pandas as pd
import re # For natural sorting

# --- Configuration ---
# IMPORTANT: Update these paths to your actual data locations!

# Set the root directory where your original raw image sequences (train/test/valid) are located.
# These files will NOT be modified.
# Double-check this path VERY carefully! It should be the folder directly containing 'train', 'test', 'valid'.
RAW_IMAGE_SEQUENCES_ROOT = r'E:/User/my work/Summer project/Code/data_preprocessing/Dataset'

# Set the root directory where the RENAMED image sequences will be saved.
# A new identical directory structure will be created here.
RENAMED_IMAGE_SEQUENCES_ROOT = r'E:/User/my work/Summer project/Code/data_preprocessing/RenamedDataset'

# Path to your main labeled_data.csv file.
# This CSV contains the original complex filenames and their labels.
YOUR_LABELED_DATA_CSV = r'E:/User/my work/Summer project/Code/labeled_data.csv'


# --- Main Renaming Function ---
def rename_frames_in_sequences_non_in_place(raw_root_dir, renamed_root_dir, label_csv_path):
    """
    Iterates through train/test/valid splits in raw_root_dir.
    Copies and renames image files to a sequential numerical format (e.g., frame_00001.jpg)
    into a new directory structure at renamed_root_dir.
    Also generates a CSV for each clip mapping new filenames to their original filenames.

    Args:
        raw_root_dir (str): Path to the root directory containing original train/test/valid splits.
        renamed_root_dir (str): Path to the new root directory for renamed images.
        label_csv_path (str): Path to your main labeled_data.csv file.
    """
    # Ensure raw_root_dir exists
    if not os.path.exists(raw_root_dir):
        print(f"Error: Raw root directory '{raw_root_dir}' does not exist. Please check the path.")
        return

    # Ensure renamed_root_dir exists (or create it)
    os.makedirs(renamed_root_dir, exist_ok=True)

    print(f"Starting non-in-place frame renaming process from: {raw_root_dir}")
    print(f"Renamed frames will be saved to: {renamed_root_dir}")

    # Load the main labeled_data.csv once (optional, but good for context)
    try:
        df_labels_master = pd.read_csv(label_csv_path)
        df_labels_master['filename'] = df_labels_master['filename'].astype(str).str.strip()
        print(f"Loaded master labels from {label_csv_path}. Total labeled entries: {len(df_labels_master)}")
    except FileNotFoundError:
        print(f"Warning: labeled_data.csv not found at {label_csv_path}. This might be an issue for later steps if labels are needed.")
        df_labels_master = pd.DataFrame(columns=['filename', 'label']) # Create empty dataframe
    except Exception as e:
        print(f"Error loading labeled_data.csv: {e}. Please check the CSV path and format.")
        df_labels_master = pd.DataFrame(columns=['filename', 'label'])


    # Helper function for natural sorting
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    # Iterate through train, test, valid splits
    for split in ['train', 'test', 'valid']:
        split_raw_path = os.path.join(raw_root_dir, split)
        split_renamed_path = os.path.join(renamed_root_dir, split)

        if not os.path.exists(split_raw_path):
            print(f"Warning: Raw path for {split} split '{split_raw_path}' not found. Skipping {split} split.")
            continue

        os.makedirs(split_renamed_path, exist_ok=True) # Ensure output split folder exists
        print(f"\n--- Processing {split} split ---")
        
        video_clip_dirs = [d for d in os.listdir(split_raw_path) if os.path.isdir(os.path.join(split_raw_path, d))]
        
        if not video_clip_dirs:
            print(f"No video clip directories found in {split_raw_path}. Skipping for this split.")
            continue

        for clip_name in video_clip_dirs:
            clip_raw_full_path = os.path.join(split_raw_path, clip_name)
            clip_renamed_full_path = os.path.join(split_renamed_path, clip_name)
            os.makedirs(clip_renamed_full_path, exist_ok=True) # Ensure output clip folder exists

            print(f"  Processing clip: {clip_name}")

            # Get image files, naturally sorted
            image_files = [f for f in os.listdir(clip_raw_full_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            image_files.sort(key=natural_sort_key) 
            
            if not image_files:
                print(f"    No image files found in {clip_raw_full_path}. Skipping clip.")
                continue

            # List to store mapping for this clip: (new_filename, original_filename)
            clip_mapping_data = []

            for i, original_name in enumerate(image_files):
                _, ext = os.path.splitext(original_name)
                new_name = f"frame_{i+1:05d}{ext}" # Renames to frame_00001.jpg, frame_00002.jpg etc.

                src_path = os.path.join(clip_raw_full_path, original_name)
                dst_path = os.path.join(clip_renamed_full_path, new_name)
                
                # --- DEBUGGING / ERROR CHECKING ADDED HERE ---
                print(f"    Attempting to copy from: {src_path}")
                if not os.path.exists(src_path):
                    print(f"    ERROR: Source file DOES NOT EXIST: {src_path}. Skipping this file.")
                    continue # Skip to the next file if this one isn't found
                # --- END DEBUGGING ---

                try:
                    shutil.copy(src_path, dst_path) # Use copy instead of move
                    clip_mapping_data.append({
                        'new_filename': new_name,
                        'original_filename': original_name,
                        'clip_name': clip_name, # Add clip_name for convenience
                        'split': split # Add split for full context
                    })
                except Exception as e:
                    print(f"      Error copying {original_name} to {new_name}: {e}")
            
            # Save the mapping CSV for this clip
            if clip_mapping_data:
                df_clip_map = pd.DataFrame(clip_mapping_data)
                # Save this mapping CSV inside the NEW clip's folder
                map_csv_path = os.path.join(clip_renamed_full_path, f"{clip_name}_frame_map.csv")
                df_clip_map.to_csv(map_csv_path, index=False)
                print(f"  Generated frame mapping CSV for {clip_name}: {map_csv_path}")
            else:
                print(f"  No frames processed for {clip_name}, no mapping CSV generated.")

            print(f"  Finished processing for clip: {clip_name}. Total {len(image_files)} frames processed.")

    print("\n--- All frames processed and mappings generated successfully! ---")
    print(f"Original frames are untouched in '{raw_root_dir}'.")
    print(f"Renamed frames and mappings are saved to '{renamed_root_dir}'.")


# --- Run the script ---
if __name__ == "__main__":
    # IMPORTANT: Double-check and update these paths to your actual data locations!
    # Ensure RAW_IMAGE_SEQUENCES_ROOT contains 'train', 'test', 'valid' splits,
    # each containing folders for individual video clips.

    rename_frames_in_sequences_non_in_place(
        RAW_IMAGE_SEQUENCES_ROOT,
        RENAMED_IMAGE_SEQUENCES_ROOT,
        YOUR_LABELED_DATA_CSV
    )


