import os
import pandas as pd
import numpy as np
import shutil
import re 
import json 

# --- Configuration Paths ---
PROCESSED_DATA_ROOT = r'E:/User/my work/Summer project/Code/processed_data_output'
YOUR_LABEL_CSVS = [
    r'E:/User/my work/Summer project/Code/labeled_data.csv',       
    r'E:/User/my work/Summer project/Code/labeled_data_test.csv',  
    r'E:/User/my work/Summer project/Code/labeled_data_valid.csv' 
]
RENAMED_DATASET_ROOT = r'E:/User/my work/Summer project/Code/data_preprocessing/RenamedDataset'
FINAL_DATASET_ROOT = r'E:/User/my work/Summer project/Code/final_dataset_for_cnn_lstm'

# --- Sequence Configuration ---
SEQUENCE_LENGTH = 10 
RANDOM_SEED = 42 

def extract_video_id(original_filename):
    match = re.match(r'(.+?)(-\d{4,5}_jpg|\.mp4)', original_filename)
    if match:
        return match.group(1).replace('_mp4', '').strip()
    match_rf = re.match(r'(.+?)\.rf\.', original_filename)
    if match_rf:
        return match_rf.group(1).replace('_mp4', '').strip()
    return original_filename.split('-')[0].split('.rf.')[0].replace('_mp4', '').strip()

def create_dataset_sequences(processed_data_root, label_csv_paths, renamed_data_root, final_dataset_root):
    if os.path.exists(final_dataset_root):
        print(f"Clearing existing {final_dataset_root}...")
        shutil.rmtree(final_dataset_root)
    os.makedirs(final_dataset_root, exist_ok=True)

    print(f"Starting dataset sequence creation. Output will be in: {final_dataset_root}")

    all_labels_dfs = []
    for csv_path in label_csv_paths:
        try:
            df = pd.read_csv(csv_path)
            df['filename'] = df['filename'].astype(str).str.strip()
            df['label'] = df['label'].astype(str).str.strip()
            all_labels_dfs.append(df)
            print(f"Loaded labels from {csv_path}. Entries: {len(df)}")
        except FileNotFoundError:
            print(f"Warning: Label CSV not found at {csv_path}. Skipping.")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}. Skipping.")

    if not all_labels_dfs:
        print("Error: No label CSVs found or loaded. Cannot proceed.")
        return

    df_labels_master = pd.concat(all_labels_dfs, ignore_index=True)
    df_labels_master.drop_duplicates(subset=['filename'], inplace=True) 
    
    initial_master_labels_count = len(df_labels_master)
    df_labels_master.dropna(subset=['label'], inplace=True)
    # --- FIX: Also remove rows where 'label' is literally the string "nan" ---
    df_labels_master = df_labels_master[df_labels_master['label'].astype(str).str.lower() != 'nan']
    # --- END FIX ---
    if len(df_labels_master) < initial_master_labels_count:
        print(f"Note: Dropped {initial_master_labels_count - len(df_labels_master)} entries from master labels due to missing or 'nan' values.")

    label_lookup = df_labels_master.set_index('filename')['label'].to_dict()
    print(f"Combined master label lookup for {len(label_lookup)} unique original filenames.")

    unique_string_labels = sorted(df_labels_master['label'].unique()) 
    string_to_int_label_map = {label: i for i, label in enumerate(unique_string_labels)}
    int_to_string_label_map = {i: label for label, i in string_to_int_label_map.items()}
    print(f"Mapped {len(unique_string_labels)} string labels to integers: {string_to_int_label_map}")


    all_clip_infos = []

    split_name_map = {'train': 'train', 'test': 'test', 'valid': 'val'}

    for original_split_type in ['train', 'test', 'valid']:
        current_split_name = split_name_map[original_split_type] 
        
        split_processed_path = os.path.join(processed_data_root, original_split_type)
        split_renamed_path = os.path.join(renamed_data_root, original_split_type)
        
        if not os.path.exists(split_processed_path):
            print(f"Warning: Processed data path for {original_split_type} '{split_processed_path}' not found. Skipping.")
            continue
        if not os.path.exists(split_renamed_path):
            print(f"Warning: Renamed data path for {original_split_type} '{split_renamed_path}' not found. Cannot load _frame_map.csv. Skipping.")
            continue

        processed_clip_dirs = sorted([d for d in os.listdir(split_processed_path) if os.path.isdir(os.path.join(split_processed_path, d)) and d.endswith('_annotated_images')])
        
        if not processed_clip_dirs:
            print(f"No annotated image directories found in {split_processed_path}. Skipping {original_split_type} split.")
            continue

        print(f"\n--- Collecting clip info for original {original_split_type} split ---")
        for annotated_images_dir_name in processed_clip_dirs:
            clip_name = annotated_images_dir_name.replace('_annotated_images', '')
            clip_processed_images_path = os.path.join(split_processed_path, annotated_images_dir_name)
            clip_features_csv_path = os.path.join(split_processed_path, f"{clip_name}_features.csv")
            
            clip_frame_map_csv_path = os.path.join(split_renamed_path, clip_name, f"{clip_name}_frame_map.csv")


            if not os.path.exists(clip_features_csv_path):
                print(f"  Warning: Skipping clip '{clip_name}' (features CSV missing at {clip_features_csv_path}).")
                continue
            if not os.path.exists(clip_frame_map_csv_path):
                print(f"  Warning: Skipping clip '{clip_name}' (frame map CSV missing at {clip_frame_map_csv_path}).")
                continue
            
            df_features = pd.read_csv(clip_features_csv_path)
            df_frame_map = pd.read_csv(clip_frame_map_csv_path)
            frame_map_lookup = df_frame_map.set_index('new_filename')['original_filename'].to_dict()

            df_features['long_original_filename'] = df_features['original_image_filename'].map(frame_map_lookup)

            df_features['label_string'] = df_features['long_original_filename'].map(label_lookup)
            df_features['label'] = df_features['label_string'].map(string_to_int_label_map)


            initial_frames = len(df_features)
            df_features.dropna(subset=['label'], inplace=True)
            df_features = df_features[df_features['label_string'].astype(str).str.lower() != 'nan'] # Filter out "nan" string labels in df_features too
            if len(df_features) < initial_frames:
                print(f"  Note: Dropped {initial_frames - len(df_features)} frames from '{clip_name}' due to missing or 'nan' labels after linking.")


            if df_features.empty:
                print(f"  Skipping clip '{clip_name}': No labeled frames found after linking and filtering.")
                continue

            video_id = extract_video_id(df_features['long_original_filename'].iloc[0])

            all_clip_infos.append({
                'video_id': video_id,
                'clip_name': clip_name,
                'df_features': df_features,
                'clip_processed_images_path': clip_processed_images_path,
                'final_split': current_split_name 
            })
            print(f"  Collected clip '{clip_name}' (Video ID: {video_id}) with {len(df_features)} labeled frames, assigned to '{current_split_name}' split.")

    if not all_clip_infos:
        print("Error: No clips found with labeled frames to process after all linking and filtering. Exiting.")
        return
    
    sequence_counter = {'train': 0, 'val': 0, 'test': 0}
    all_sequence_metadata = []

    for clip_info in all_clip_infos:
        video_id = clip_info['video_id']
        clip_name = clip_info['clip_name']
        df_features_clip = clip_info['df_features'].sort_values(by='frame_index').reset_index(drop=True)
        clip_processed_images_path = clip_info['clip_processed_images_path']
        
        current_final_split_type = clip_info['final_split'] 

        num_frames = len(df_features_clip)
        
        for i in range(0, num_frames - SEQUENCE_LENGTH + 1):
            sequence_df = df_features_clip.iloc[i : i + SEQUENCE_LENGTH]
            
            unique_labels_in_sequence = sequence_df['label'].unique()
            if len(unique_labels_in_sequence) != 1 or pd.isna(unique_labels_in_sequence[0]):
                continue

            sequence_label_int = int(unique_labels_in_sequence[0])
            sequence_label_string = int_to_string_label_map[sequence_label_int]


            final_sequence_dir = os.path.join(final_dataset_root, current_final_split_type, sequence_label_string, f"sequence_{sequence_counter[current_final_split_type]:05d}")
            os.makedirs(final_sequence_dir, exist_ok=True)
            os.makedirs(os.path.join(final_sequence_dir, 'images'), exist_ok=True)

            numerical_columns = [col for col in sequence_df.columns if col not in ['clip_id', 'frame_index', 'original_image_filename', 'long_original_filename', 'label_string', 'label', 'annotated_image_path']]
            sequence_numerical_features = sequence_df[numerical_columns].values
            np.save(os.path.join(final_sequence_dir, 'features.npy'), sequence_numerical_features)

            for frame_new_name in sequence_df['original_image_filename']:
                annotated_img_src_path = os.path.join(clip_processed_images_path, f"{os.path.splitext(frame_new_name)[0]}_annotated.jpg")
                annotated_img_dst_path = os.path.join(final_sequence_dir, 'images', frame_new_name)
                
                if os.path.exists(annotated_img_src_path):
                    shutil.copy(annotated_img_src_path, annotated_img_dst_path)
                else:
                    print(f"      Warning: Annotated image not found at {annotated_img_src_path}. Skipping copy for this frame.")


            all_sequence_metadata.append({
                'final_split': current_final_split_type,
                'class_label_string': sequence_label_string,
                'class_label_int': sequence_label_int,
                'sequence_id_in_split': f"sequence_{sequence_counter[current_final_split_type]:05d}",
                'source_video_id': video_id,
                'source_clip_name': clip_name,
                'start_frame_index': i,
                'end_frame_index': i + SEQUENCE_LENGTH - 1,
                'path': os.path.relpath(final_sequence_dir, start=final_dataset_root)
            })

            sequence_counter[current_final_split_type] += 1

    print(f"\n--- Final Dataset Creation Complete! ---")
    print(f"Total sequences generated:")
    for split_type, count in sequence_counter.items():
        print(f"  {split_type.capitalize()}: {count} sequences")
    print(f"Dataset saved to: {final_dataset_root}")
    print(f"String to Int Class Mapping: {string_to_int_label_map}")

    metadata_path = os.path.join(final_dataset_root, 'dataset_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(all_sequence_metadata, f, indent=4)
    print(f"Dataset metadata saved to: {metadata_path}")


# --- Run the script ---
if __name__ == "__main__":
    create_dataset_sequences(
        PROCESSED_DATA_ROOT,
        YOUR_LABEL_CSVS,
        RENAMED_DATASET_ROOT,
        FINAL_DATASET_ROOT
    )

