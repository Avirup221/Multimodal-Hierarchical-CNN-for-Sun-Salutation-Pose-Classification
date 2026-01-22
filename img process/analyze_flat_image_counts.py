import os
import shutil
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import re
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R # Used for some feature calculations

# --- Configuration Paths ---
# IMPORTANT: Update these paths to your actual data locations!

# Input: Root directory of your renamed image dataset (output from Frame_Renaming_Staged.py)
RENAMED_DATASET_ROOT = r'E:/User/my work/Summer project/Code/data_preprocessing/RenamedDataset'

# Input: List of paths to your label CSV files.
YOUR_LABEL_CSVS = [
    r'E:/User/my work/Summer project/Code/labeled_data.csv',       # Your original train labels
    r'E:/User/my work/Summer project/Code/labeled_data_test.csv',  # Your newly created test labels
    r'E:/User/my work/Summer project/Code/labeled_data_valid.csv' # Your newly created validation labels
]

# Output: The new root directory for the flattened, single-image dataset.
# This is where your 'train', 'test', 'valid' folders will be created,
# and inside them: 'class_name/image.jpg'.
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'

# --- MediaPipe Pose configuration ---
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# --- Feature Extraction Helper Functions (Copied from processing_image_sequence.py) ---

def calculate_angle(p1, p2, p3):
    """Calculates angle (in degrees) between three 3D points."""
    a = np.array(p1)
    b = np.array(p2) # Mid point
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 3D points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_landmark_coords_norm(landmarks):
    """Extracts normalized (x,y,z) and visibility for all 33 landmarks."""
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z, lm.visibility]) # Normalized x, y, z and visibility
    return np.array(coords)

def get_landmark_coords_pixel(landmarks, img_w, img_h):
    """Extracts pixel (x,y) and scaled z, visibility."""
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x * img_w, lm.y * img_h, lm.z * img_w, lm.visibility]) # Pixel x, y, scaled z
    return np.array(coords)

def calculate_torso_angle_vertical(landmarks):
    """Calculates angle of torso relative to vertical axis (from hip to shoulder)."""
    left_hip = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                         landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                         landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z])
    right_hip = np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                          landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                          landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z])
    mid_hip = (left_hip + right_hip) / 2

    left_shoulder = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                              landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                              landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z])
    right_shoulder = np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                               landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                               landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z])
    mid_shoulder = (left_shoulder + right_shoulder) / 2

    torso_vector = mid_shoulder - mid_hip
    vertical_vector = np.array([0, 1, 0]) 

    torso_vector_2d = torso_vector[:2]
    vertical_vector_2d = np.array([0, 1])

    angle_rad = np.arctan2(vertical_vector_2d[1], vertical_vector_2d[0]) - np.arctan2(torso_vector_2d[1], torso_vector_2d[0])
    angle_deg = np.degrees(angle_rad)
    
    angle_deg = np.abs(angle_deg)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
        
    return angle_deg

def calculate_torso_alignment_horizontal(landmarks):
    """Calculates horizontal alignment (e.g., angle of line between shoulders and hips)."""
    left_hip = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y])
    right_hip = np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y])
    left_shoulder = np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y])
    right_shoulder = np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y])

    shoulder_vector = right_shoulder - left_shoulder
    hip_vector = right_hip - left_hip

    shoulder_angle = np.degrees(np.arctan2(shoulder_vector[1], shoulder_vector[0]))
    hip_angle = np.degrees(np.arctan2(hip_vector[1], hip_vector[0]))

    alignment_diff = np.abs(shoulder_angle - hip_angle)
    if alignment_diff > 180:
        alignment_diff = 360 - alignment_diff
    return alignment_diff

# Define the full list of 575 feature column names for consistent output
# This list MUST match the number of features your model expects (575)
ALL_FEATURE_COLUMNS = []
for i in range(mp_pose.PoseLandmark.LEFT_SHOULDER.value + 1): # Nose to R_FOOT_INDEX (33 landmarks)
    ALL_FEATURE_COLUMNS.extend([f'LM{i}_norm_x', f'LM{i}_norm_y', f'LM{i}_norm_z', f'LM{i}_visibility'])
ALL_FEATURE_COLUMNS.extend([
    'LEFT_ELBOW_ANGLE', 'RIGHT_ELBOW_ANGLE', 'LEFT_SHOULDER_ANGLE', 'RIGHT_SHOULDER_ANGLE',
    'LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 'LEFT_HIP_ANGLE', 'RIGHT_HIP_ANGLE',
    'TORSO_VERTICAL_ANGLE', 'TORSO_HORIZONTAL_ALIGNMENT'
])
ALL_FEATURE_COLUMNS.extend([
    'DIST_LR_WRIST_NORM', 'DIST_LR_ANKLE_NORM', 'DIST_L_WRIST_HIP_NORM'
])
for i in range(mp_pose.PoseLandmark.LEFT_SHOULDER.value + 1):
    ALL_FEATURE_COLUMNS.extend([f'LM{i}_rel_x_norm', f'LM{i}_rel_y_norm', f'LM{i}_rel_z_norm'])
for i in range(mp_pose.PoseLandmark.LEFT_SHOULDER.value + 1):
    ALL_FEATURE_COLUMNS.extend([
        f'LM{i}_vx_px', f'LM{i}_vy_px', f'LM{i}_vz_px', 
        f'LM{i}_ax_px', f'LM{i}_ay_px', f'LM{i}_az_px'  
    ])
ALL_FEATURE_COLUMNS.append('TORSO_VAR_XY_RATIO')

# --- Function to extract video ID from your complex original_filename ---
def extract_video_id(original_filename):
    match = re.match(r'(.+?)(-\d{4,5}_jpg|\.mp4)', original_filename)
    if match:
        return match.group(1).replace('_mp4', '').strip()
    match_rf = re.match(r'(.+?)\.rf\.', original_filename)
    if match_rf:
        return match_rf.group(1).replace('_mp4', '').strip()
    return original_filename.split('-')[0].split('.rf.')[0].replace('_mp4', '').strip()


def prepare_still_image_dataset(renamed_dataset_root, label_csv_paths, still_image_dataset_root):
    # Clean output directory
    if os.path.exists(still_image_dataset_root):
        print(f"Clearing existing {still_image_dataset_root}...")
        shutil.rmtree(still_image_dataset_root)
    os.makedirs(still_image_dataset_root, exist_ok=True)

    print(f"Starting still image dataset preparation. Output will be in: {still_image_dataset_root}")

    # 1. Load all label CSVs and combine them into one master lookup
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
    df_labels_master = df_labels_master[df_labels_master['label'].astype(str).str.lower() != 'nan'] # Filter out "nan" string labels
    
    if len(df_labels_master) < initial_master_labels_count:
        print(f"Note: Dropped {initial_master_labels_count - len(df_labels_master)} entries from master labels due to missing or 'nan' values.")

    label_lookup = df_labels_master.set_index('filename')['label'].to_dict()
    print(f"Combined master label lookup for {len(label_lookup)} unique original filenames.")

    unique_string_labels = sorted(df_labels_master['label'].unique()) 
    string_to_int_label_map = {label: i for i, label in enumerate(unique_string_labels)}
    int_to_string_label_map = {i: label for label, i in string_to_int_label_map.items()}
    print(f"Mapped {len(unique_string_labels)} string labels to integers: {string_to_int_label_map}")

    # Initialize counters for final dataset
    final_image_counts = {'train': 0, 'test': 0, 'valid': 0}

    # Iterate through the original splits (train, test, valid) in RenamedDataset
    for split_type in ['train', 'test', 'valid']:
        input_split_dir = os.path.join(renamed_dataset_root, split_type)
        output_split_dir = os.path.join(still_image_dataset_root, split_type) # Output for this split
        os.makedirs(output_split_dir, exist_ok=True) # Ensure split directory exists

        if not os.path.exists(input_split_dir):
            print(f"Warning: Input directory for {split_type} '{input_split_dir}' not found. Skipping.")
            continue

        clip_dirs = [d for d in os.listdir(input_split_dir) if os.path.isdir(os.path.join(input_split_dir, d))]
        
        print(f"\n--- Processing images for {split_type} split ---")
        for clip_name in tqdm(clip_dirs, desc=f"Overall {split_type} clips"):
            current_clip_images_dir = os.path.join(input_split_dir, clip_name)
            clip_frame_map_csv_path = os.path.join(current_clip_images_dir, f"{clip_name}_frame_map.csv")

            if not os.path.exists(clip_frame_map_csv_path):
                print(f"  Warning: Skipping clip '{clip_name}' (frame map CSV missing at {clip_frame_map_csv_path}). Cannot link images to original filenames.")
                continue
            
            df_frame_map = pd.read_csv(clip_frame_map_csv_path)
            frame_map_lookup = df_frame_map.set_index('new_filename')['original_filename'].to_dict()

            image_files = sorted([f for f in os.listdir(current_clip_images_dir) if f.endswith(('.jpg', '.png'))])
            
            # Store features for all frames in this clip (for dynamic features)
            prev_landmark_px_coords = None
            prev_velocity_px = None

            for i, img_file in enumerate(image_files):
                original_filename = frame_map_lookup.get(img_file)
                if original_filename is None:
                    # print(f"    Warning: No original filename mapping for {img_file} in {clip_name}. Skipping.")
                    continue
                
                # Get label for this specific frame
                label_string = label_lookup.get(original_filename)

                # Only process and copy if a valid, non-NaN label is found
                if label_string is not None and str(label_string).lower() != 'nan':
                    class_label_int = string_to_int_label_map[label_string]
                    class_label_string = label_string # Use the string label for folder name

                    src_img_path = os.path.join(current_clip_images_dir, img_file)
                    image = cv2.imread(src_img_path)
                    if image is None:
                        print(f"Warning: Could not read image {src_img_path}. Skipping.")
                        continue

                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    img_h, img_w, _ = image.shape

                    results = POSE.process(image_rgb)
                    
                    # --- Generate Features for ALL frames, fill with NaN/0 if no pose ---
                    current_frame_features_dict = {} # Use dict to build features

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks
                        # Raw Normalized 3D Coordinates & Visibility (33*4 = 132 features)
                        lm_coords_norm = get_landmark_coords_norm(landmarks)
                        for j in range(33):
                            current_frame_features_dict[f'LM{j}_norm_x'] = lm_coords_norm[j*4] if lm_coords_norm[j*4+3] > 0.65 else np.nan
                            current_frame_features_dict[f'LM{j}_norm_y'] = lm_coords_norm[j*4+1] if lm_coords_norm[j*4+3] > 0.65 else np.nan
                            current_frame_features_dict[f'LM{j}_norm_z'] = lm_coords_norm[j*4+2] if lm_coords_norm[j*4+3] > 0.65 else np.nan
                            current_frame_features_dict[f'LM{j}_visibility'] = lm_coords_norm[j*4+3] # Always keep visibility
                        
                        lm_coords_px = get_landmark_coords_pixel(landmarks, img_w, img_h)

                        # Derived Features: Angles
                        points = {}
                        for j, lm_enum in enumerate(mp_pose.PoseLandmark):
                            points[lm_enum.name] = np.array([landmarks.landmark[j].x, landmarks.landmark[j].y, landmarks.landmark[j].z])
                        
                        angles_to_calc = {
                            'LEFT_ELBOW_ANGLE': (points['LEFT_SHOULDER'], points['LEFT_ELBOW'], points['LEFT_WRIST']),
                            'RIGHT_ELBOW_ANGLE': (points['RIGHT_SHOULDER'], points['RIGHT_ELBOW'], points['RIGHT_WRIST']),
                            'LEFT_SHOULDER_ANGLE': (points['LEFT_HIP'], points['LEFT_SHOULDER'], points['LEFT_ELBOW']),
                            'RIGHT_SHOULDER_ANGLE': (points['RIGHT_HIP'], points['RIGHT_SHOULDER'], points['RIGHT_ELBOW']),
                            'LEFT_KNEE_ANGLE': (points['LEFT_HIP'], points['LEFT_KNEE'], points['LEFT_ANKLE']),
                            'RIGHT_KNEE_ANGLE': (points['RIGHT_HIP'], points['RIGHT_KNEE'], points['RIGHT_ANKLE']),
                            'LEFT_HIP_ANGLE': (points['LEFT_SHOULDER'], points['LEFT_HIP'], points['LEFT_KNEE']),
                            'RIGHT_HIP_ANGLE': (points['RIGHT_SHOULDER'], points['RIGHT_HIP'], points['RIGHT_KNEE'])
                        }
                        
                        for angle_name, (p1, p2, p3) in angles_to_calc.items():
                            try: current_frame_features_dict[angle_name] = calculate_angle(p1, p2, p3)
                            except: current_frame_features_dict[angle_name] = np.nan

                        try:
                            current_frame_features_dict['TORSO_VERTICAL_ANGLE'] = calculate_torso_angle_vertical(landmarks)
                            current_frame_features_dict['TORSO_HORIZONTAL_ALIGNMENT'] = calculate_torso_alignment_horizontal(landmarks)
                        except:
                            current_frame_features_dict['TORSO_VERTICAL_ANGLE'] = np.nan
                            current_frame_features_dict['TORSO_HORIZONTAL_ALIGNMENT'] = np.nan

                        shoulder_width = calculate_distance(points['LEFT_SHOULDER'], points['RIGHT_SHOULDER'])
                        hip_width = calculate_distance(points['LEFT_HIP'], points['RIGHT_HIP'])
                        body_scale = np.mean([shoulder_width, hip_width]) if (shoulder_width > 0 and hip_width > 0) else 1.0 
                        if body_scale == 0: body_scale = 1.0 

                        if body_scale > 0.05: 
                            try: current_frame_features_dict['DIST_LR_WRIST_NORM'] = calculate_distance(points['LEFT_WRIST'], points['RIGHT_WRIST']) / body_scale
                            except: current_frame_features_dict['DIST_LR_WRIST_NORM'] = np.nan
                            try: current_frame_features_dict['DIST_LR_ANKLE_NORM'] = calculate_distance(points['LEFT_ANKLE'], points['RIGHT_ANKLE']) / body_scale
                            except: current_frame_features_dict['DIST_LR_ANKLE_NORM'] = np.nan
                            try: current_frame_features_dict['DIST_L_WRIST_HIP_NORM'] = calculate_distance(points['LEFT_WRIST'], points['LEFT_HIP']) / body_scale
                            except: current_frame_features_dict['DIST_L_WRIST_HIP_NORM'] = np.nan
                        else:
                            current_frame_features_dict['DIST_LR_WRIST_NORM'] = np.nan
                            current_frame_features_dict['DIST_LR_ANKLE_NORM'] = np.nan
                            current_frame_features_dict['DIST_L_WRIST_HIP_NORM'] = np.nan

                        mid_hip = (np.array([landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z]) + 
                                   np.array([landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y, landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z])) / 2

                        for j in range(33):
                            lm_x = landmarks.landmark[j].x
                            lm_y = landmarks.landmark[j].y
                            lm_z = landmarks.landmark[j].z
                            
                            if landmarks.landmark[j].visibility > 0.65 and np.linalg.norm(mid_hip) > 0: 
                                current_frame_features_dict[f'LM{j}_rel_x_norm'] = lm_x - mid_hip[0]
                                current_frame_features_dict[f'LM{j}_rel_y_norm'] = lm_y - mid_hip[1]
                                current_frame_features_dict[f'LM{j}_rel_z_norm'] = lm_z - mid_hip[2]
                            else:
                                current_frame_features_dict[f'LM{j}_rel_x_norm'] = np.nan
                                current_frame_features_dict[f'LM{j}_rel_y_norm'] = np.nan
                                current_frame_features_dict[f'LM{j}_rel_z_norm'] = np.nan

                        current_landmark_px_coords = np.array([lm_coords_px[j*4:(j+1)*4][:3] for j in range(33)]).flatten()
                        
                        velocity_px = np.full_like(current_landmark_px_coords, np.nan)
                        acceleration_px = np.full_like(current_landmark_px_coords, np.nan)

                        if prev_landmark_px_coords is not None:
                            velocity_px = current_landmark_px_coords - prev_landmark_px_coords
                            if prev_velocity_px is not None:
                                acceleration_px = velocity_px - prev_velocity_px
                        
                        for j in range(33):
                            current_frame_features_dict[f'LM{j}_vx_px'] = velocity_px[j*3] if not np.isnan(velocity_px[j*3]) else np.nan
                            current_frame_features_dict[f'LM{j}_vy_px'] = velocity_px[j*3+1] if not np.isnan(velocity_px[j*3+1]) else np.nan
                            current_frame_features_dict[f'LM{j}_vz_px'] = velocity_px[j*3+2] if not np.isnan(velocity_px[j*3+2]) else np.nan

                            current_frame_features_dict[f'LM{j}_ax_px'] = acceleration_px[j*3] if not np.isnan(acceleration_px[j*3]) else np.nan
                            current_frame_features_dict[f'LM{j}_ay_px'] = acceleration_px[j*3+1] if not np.isnan(acceleration_px[j*3+1]) else np.nan
                            current_frame_features_dict[f'LM{j}_az_px'] = acceleration_px[j*3+2] if not np.isnan(acceleration_px[j*3+2]) else np.nan

                        try:
                            torso_lms_x = [landmarks.landmark[11].x, landmarks.landmark[12].x, landmarks.landmark[23].x, landmarks.landmark[24].x]
                            torso_lms_y = [landmarks.landmark[11].y, landmarks.landmark[12].y, landmarks.landmark[23].y, landmarks.landmark[24].y]
                            
                            visible_torso_lms_x = [x for i, x in enumerate(torso_lms_x) if landmarks.landmark[[11,12,23,24][i]].visibility > 0.65]
                            visible_torso_lms_y = [y for i, y in enumerate(torso_lms_y) if landmarks.landmark[[11,12,23,24][i]].visibility > 0.65]

                            if len(visible_torso_lms_x) >= 2 and len(visible_torso_lms_y) >= 2: 
                                var_x = np.var(visible_torso_lms_x)
                                var_y = np.var(visible_torso_lms_y)
                                current_frame_features_dict['TORSO_VAR_XY_RATIO'] = var_x / var_y if var_y != 0 else np.nan 
                            else:
                                current_frame_features_dict['TORSO_VAR_XY_RATIO'] = np.nan
                        except:
                            current_frame_features_dict['TORSO_VAR_XY_RATIO'] = np.nan

                    else: # No pose landmarks detected
                        for col in ALL_FEATURE_COLUMNS:
                            current_frame_features_dict[col] = np.nan # Fill all with NaN
                        for j in range(33):
                            current_frame_features_dict[f'LM{j}_visibility'] = 0.0 # Explicitly zero visibility if no pose

                    # Update dynamic feature trackers for the next frame
                    prev_landmark_px_coords = current_landmark_px_coords if results.pose_landmarks else None
                    prev_velocity_px = velocity_px if results.pose_landmarks else None

                    # --- Copy Image & Save Features ---
                    dest_class_dir = os.path.join(output_split_dir, class_label_string)
                    os.makedirs(dest_class_dir, exist_ok=True)
                    
                    # To ensure unique names, prepend clip_name and frame_index
                    unique_img_name = f"{clip_name}_frame_{i:05d}_{img_file}"
                    dest_img_path = os.path.join(dest_class_dir, unique_img_name)

                    shutil.copy2(src_img_path, dest_img_path)
                    
                    # Save numerical features as .npy alongside the image
                    # Convert dict to array in the correct order
                    numerical_features_array = np.array([current_frame_features_dict.get(col, np.nan) for col in ALL_FEATURE_COLUMNS], dtype=np.float32)
                    np.save(os.path.join(dest_class_dir, f"{os.path.splitext(unique_img_name)[0]}.npy"), numerical_features_array)

                    final_image_counts[split_type] += 1
                # else:
                #     print(f"    Skipping {img_file} in {clip_name} due to missing or 'nan' label.")
    
    print(f"\n--- Still Image Dataset Preparation Complete! ---")
    print(f"Total images prepared for classification:")
    for split_type, count in final_image_counts.items():
        print(f"  {split_type.capitalize()} Split: {count} images")
    print(f"Dataset saved to: {STILL_IMAGE_DATASET_ROOT}")

# --- Run the script ---
if __name__ == "__main__":
    prepare_still_image_dataset(
        RENAMED_DATASET_ROOT,
        YOUR_LABEL_CSVS,
        STILL_IMAGE_DATASET_ROOT
    )

