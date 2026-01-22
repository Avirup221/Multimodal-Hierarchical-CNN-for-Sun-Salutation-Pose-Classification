import os
import shutil
import json

# --- Configuration Paths ---
# IMPORTANT: Update these paths!

# Input: Root directory of your previously created sequential dataset.
# This is where 'dataset_metadata.json' and 'train', 'val', 'test' folders are.
FINAL_DATASET_ROOT = r'E:/User/my work/Summer project/Code/final_dataset_for_cnn_lstm'

# Output: The new root directory for the flattened, single-image dataset.
# This is where your 'train', 'val', 'test' folders will be created,
# and inside them: 'class_name/image.jpg'.
CLASS_IMAGES_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset'

def reorganize_dataset_to_flat_images(final_dataset_root, class_images_root):
    """
    Reorganizes the sequential dataset into a flat directory structure
    for single-image classification: CLASS_IMAGES_ROOT/split/class_name/image_filename.jpg
    """
    # Clean output directory
    if os.path.exists(class_images_root):
        print(f"Clearing existing {class_images_root}...")
        shutil.rmtree(class_images_root)
    os.makedirs(class_images_root, exist_ok=True)

    metadata_path = os.path.join(final_dataset_root, 'dataset_metadata.json')

    if not os.path.exists(metadata_path):
        print(f"Error: dataset_metadata.json not found at {metadata_path}. Please run create_sequential_dataset.py first.")
        return

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}. Total sequences in metadata: {len(metadata)}")
    except Exception as e:
        print(f"Error loading metadata from {metadata_path}: {e}")
        return

    print(f"Starting image reorganization to: {class_images_root}")
    copied_count = 0

    for seq_info in metadata:
        final_split = seq_info['final_split']
        class_label_string = seq_info['class_label_string']
        
        # Path to the source images for this sequence within the sequential dataset
        # Construct the full path based on the 'path' in metadata which is relative to final_dataset_root
        source_seq_images_dir = os.path.join(final_dataset_root, seq_info['path'], 'images')

        if not os.path.exists(source_seq_images_dir):
            print(f"Warning: Source images directory not found: {source_seq_images_dir}. Skipping sequence.")
            continue
        
        # Destination directory for this class within the new flattened structure
        dest_class_dir = os.path.join(class_images_root, final_split, class_label_string)
        os.makedirs(dest_class_dir, exist_ok=True)

        # Copy each image from the sequence to the new flat class directory
        image_filenames = sorted([f for f in os.listdir(source_seq_images_dir) if f.endswith(('.jpg', '.png'))])
        
        for img_name in image_filenames:
            src_img_path = os.path.join(source_seq_images_dir, img_name)
            
            # To ensure unique names, we can append sequence_id if necessary,
            # but for now, frame_000x.jpg from different sequences could clash.
            # Let's ensure uniqueness by prepending sequence_id or using a hash.
            # For simplicity, let's prepend the sequence ID to the filename:
            unique_img_name = f"{seq_info['sequence_id_in_split']}_{img_name}"
            dest_img_path = os.path.join(dest_class_dir, unique_img_name)

            # Only copy if the file does not already exist (to handle potential duplicate frame numbers
            # from different sequences, although sequence_id_in_split should ensure uniqueness here)
            if not os.path.exists(dest_img_path):
                shutil.copy2(src_img_path, dest_img_path)
                copied_count += 1
            # else:
            #     print(f"  Skipping existing image: {unique_img_name}")

    print(f"\n--- Image Reorganization Complete! ---")
    print(f"Copied {copied_count} individual images to {class_images_root}")

# --- Run the script ---
if __name__ == "__main__":
    reorganize_dataset_to_flat_images(FINAL_DATASET_ROOT, CLASS_IMAGES_ROOT)

