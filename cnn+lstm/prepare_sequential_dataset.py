import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import re
import json
import warnings

# --- CONFIGURATION ---
# 1. Input: Your Augmented Images
AUGMENTED_DATASET_ROOT = "/home/avirupd/summer_project/zero123_augmentation/ProcessedData/FINAL_AUGMENTED_DATASET"

# 2. Input: Your Original Flat Dataset (to get the .npy files)
FLAT_DATASET_ROOT = "/home/avirupd/summer_project/flat_image_dataset_final"

# 3. Output: Where to save the new sequences
OUTPUT_DATA_PATH = "/home/avirupd/summer_project/augmented_sequential_dataset_seq4"

# --- Parameters ---
SEQ_LEN = 4
STRIDE = 2
IMAGE_SIZE = (224, 224)
# --------------------

warnings.simplefilter(action='ignore', category=FutureWarning)

def get_image_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_view_sequence(clip_id, view_id, frames_list, output_dir, split, class_to_idx, transform):
    """
    Processes a sequence of frames for a specific Video Clip AND specific View ID.
    """
    # Sort by frame index to ensure temporal order
    frames_list.sort(key=lambda x: x['frame_idx'])
    
    sequences_saved = 0
    
    # Sliding Window
    for i in range(0, len(frames_list) - SEQ_LEN + 1, STRIDE):
        window = frames_list[i : i + SEQ_LEN]
        
        image_tensors = []
        numerical_tensors = []
        
        # Label Strategy: Last frame label
        last_frame_data = window[-1]
        label_str = last_frame_data['label']
        
        # Clean the label for dictionary lookup
        clean_label_str = label_str.strip()
        
        if clean_label_str not in class_to_idx:
            continue
            
        seq_label_id = class_to_idx[clean_label_str]
        
        # Prepare Output Directory (Use clean label for folder name)
        class_output_dir = os.path.join(output_dir, split, clean_label_str)
        os.makedirs(class_output_dir, exist_ok=True)
        
        # Filename includes View ID
        output_filename = f"{clip_id}_view_{view_id}_seq_{i:05d}.pt"
        output_save_path = os.path.join(class_output_dir, output_filename)
        
        if os.path.exists(output_save_path):
            sequences_saved += 1
            continue
            
        try:
            for frame_data in window:
                # 1. Load Augmented Image
                img = Image.open(frame_data['img_path']).convert("RGB")
                img_tensor = transform(img)
                image_tensors.append(img_tensor)
                
                # 2. Load Original Numerical Features (.npy)
                npy_path = frame_data.get('npy_path')
                
                if not npy_path or not os.path.exists(npy_path):
                    # If missing, fill with zeros
                    num_data = np.zeros(47, dtype=np.float32)
                else:
                    num_data = np.load(npy_path).astype(np.float32)
                    
                numerical_tensors.append(torch.from_numpy(num_data).float())
                
            # Stack
            image_sequence = torch.stack(image_tensors)
            numerical_sequence = torch.stack(numerical_tensors)
            
            torch.save({
                'image_sequence': image_sequence,
                'numerical_sequence': numerical_sequence,
                'label': seq_label_id,
                'video_clip': clip_id,
                'view_id': view_id
            }, output_save_path)
            
            sequences_saved += 1
            
        except Exception as e:
            # print(f"Error saving sequence: {e}")
            continue

    return sequences_saved

def main():
    print("--- Preparing Augmented Sequential Dataset ---")
    
    if not os.path.exists(AUGMENTED_DATASET_ROOT):
        print(f"FATAL: Augmented dataset not found at {AUGMENTED_DATASET_ROOT}")
        return

    # 1. Build Class Map (from the folders in 'train')
    # We strip whitespace to ensure keys are clean
    print("Building class map...")
    raw_labels = [d for d in os.listdir(os.path.join(AUGMENTED_DATASET_ROOT, 'train')) 
                         if os.path.isdir(os.path.join(AUGMENTED_DATASET_ROOT, 'train', d))]
    
    clean_labels = sorted(list(set([l.strip() for l in raw_labels])))
    class_to_idx = {label: i for i, label in enumerate(clean_labels)}
    
    os.makedirs(OUTPUT_DATA_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_DATA_PATH, "class_to_idx.json"), 'w') as f:
        json.dump(class_to_idx, f, indent=4)
    
    transform = get_image_transform(IMAGE_SIZE)
    
    # --- FIX: Updated Regex to handle '.jpg' in the middle ---
    # Matches: video_clip_001_frame_00101.jpg_view_01.png
    # Group 1: video_clip_001
    # Group 2: 00101 (Frame Index)
    # Group 3: 01 (View Index)
    filename_pattern = re.compile(r'(video_clip_\d+)_frame_(\d+)(?:\.[a-zA-Z]+)?_view_(\d+)\.png')
    # ---------------------------------------------------------

    # 2. Process Splits
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(AUGMENTED_DATASET_ROOT, split)
        if not os.path.exists(split_path): continue
        
        print(f"\nProcessing split: {split}")
        
        # Group data: { clip_id: { view_id: [frame_data, ...] } }
        grouped_data = {}
        
        # Walk through label folders (these might be "dirty" with spaces)
        for label in tqdm(os.listdir(split_path), desc=f"Scanning labels ({split})"):
            label_path = os.path.join(split_path, label)
            if not os.path.isdir(label_path): continue
            
            # Optimization: Pre-scan NPY files for this label
            npy_lookup = {} 
            flat_label_dir = os.path.join(FLAT_DATASET_ROOT, split, label)
            
            if os.path.exists(flat_label_dir):
                for f in os.listdir(flat_label_dir):
                    if f.endswith('.npy'):
                        # Format: video_clip_001_frame_00000_frame_00001.npy
                        parts = f.split('_frame_')
                        if len(parts) >= 3:
                            c_id = parts[0] 
                            f_idx = parts[-1].replace('.npy', '')
                            # Map (clip_id, frame_index_string) -> full path
                            npy_lookup[(c_id, f_idx)] = os.path.join(flat_label_dir, f)

            # Scan images in the augmented folder
            for img_file in os.listdir(label_path):
                match = filename_pattern.match(img_file)
                if not match: 
                    # print(f"Regex failed for: {img_file}")
                    continue
                
                clip_id = match.group(1)      # video_clip_001
                frame_idx_str = match.group(2)# 00101
                view_id = match.group(3)      # 01
                frame_idx = int(frame_idx_str)
                
                # Find NPY using the original clip ID and frame index
                npy_path = npy_lookup.get((clip_id, frame_idx_str))
                
                if clip_id not in grouped_data:
                    grouped_data[clip_id] = {}
                if view_id not in grouped_data[clip_id]:
                    grouped_data[clip_id][view_id] = []
                    
                grouped_data[clip_id][view_id].append({
                    'frame_idx': frame_idx,
                    'img_path': os.path.join(label_path, img_file),
                    'npy_path': npy_path,
                    'label': label # Keep the original label string to find folder
                })
        
        # Process the groups
        total_seqs = 0
        
        for clip_id, views in tqdm(grouped_data.items(), desc="Processing Clips"):
            for view_id, frames in views.items():
                seqs = process_view_sequence(
                    clip_id, view_id, frames, 
                    OUTPUT_DATA_PATH, split, class_to_idx, transform
                )
                total_seqs += seqs
                
        print(f"  Total sequences created for {split}: {total_seqs}")

    print(f"\nDone! Dataset saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()