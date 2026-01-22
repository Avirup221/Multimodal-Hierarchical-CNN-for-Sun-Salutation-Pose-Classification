import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from collections import deque # For storing previous landmarks for velocity/acceleration

# --- Configuration ---
# IMPORTANT: Update these paths to match your system's directory structure!

# This should be the output path from your Frame_Renaming_Staged.py script.
# It should contain your 'train', 'test', 'valid' folders, which in turn contain
# your video_clip folders with 'frame_0000x.jpg' files.
RAW_IMAGE_SEQUENCES_ROOT = r'E:/User/my work/Summer project/Code/data_preprocessing/RenamedDataset'

# This is where the processed data (features CSVs and annotated images) will be saved.
PROCESSED_DATA_ROOT = r'E:/User/my work/Summer project/Code/processed_data_output'

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Constants for Feature Calculation ---
# Minimum visibility score for a landmark to be considered valid for calculations and primary drawing
MIN_LANDMARK_VISIBILITY = 0.65 

# Mapping MediaPipe landmark names to their indices for easier reference
LANDMARK_IDS = {
    "NOSE": 0, "LEFT_EYE_INNER": 1, "LEFT_EYE": 2, "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4, "RIGHT_EYE": 5, "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7, "RIGHT_EAR": 8, "MOUTH_LEFT": 9, "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12, "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14, "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17, "RIGHT_PINKY": 18, "LEFT_INDEX": 19, "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21, "RIGHT_THUMB": 22, "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26, "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29, "RIGHT_HEEL": 30, "LEFT_FOOT_INDEX": 31, "RIGHT_FOOT_INDEX": 32,
}

# Define keypoints for common angle calculations (p1, p2, p3 where p2 is the vertex)
ANGLE_DEFINITIONS = {
    "LEFT_ELBOW_ANGLE": (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_ELBOW"], LANDMARK_IDS["LEFT_WRIST"]),
    "RIGHT_ELBOW_ANGLE": (LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["RIGHT_ELBOW"], LANDMARK_IDS["RIGHT_WRIST"]),
    "LEFT_SHOULDER_ANGLE": (LANDMARK_IDS["LEFT_ELBOW"], LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_HIP"]),
    "RIGHT_SHOULDER_ANGLE": (LANDMARK_IDS["RIGHT_ELBOW"], LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["RIGHT_HIP"]),
    "LEFT_KNEE_ANGLE": (LANDMARK_IDS["LEFT_HIP"], LANDMARK_IDS["LEFT_KNEE"], LANDMARK_IDS["LEFT_ANKLE"]),
    "RIGHT_KNEE_ANGLE": (LANDMARK_IDS["RIGHT_HIP"], LANDMARK_IDS["RIGHT_KNEE"], LANDMARK_IDS["RIGHT_ANKLE"]),
    "LEFT_HIP_ANGLE": (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_HIP"], LANDMARK_IDS["LEFT_KNEE"]),
    "RIGHT_HIP_ANGLE": (LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["RIGHT_HIP"], LANDMARK_IDS["RIGHT_KNEE"]),
    "TORSO_VERTICAL_ANGLE": (LANDMARK_IDS["NOSE"], LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_HIP"]),
    "TORSO_HORIZONTAL_ALIGNMENT": (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["LEFT_HIP"]),
}

# --- Helper Functions for Feature Calculation ---

def get_landmark_coords_and_check_visibility(landmarks, idx, img_width, img_height, min_vis=MIN_LANDMARK_VISIBILITY):
    """
    Extracts pixel coordinates and visibility. Returns None if visibility is below threshold.
    """
    if landmarks.landmark[idx].visibility < min_vis:
        return None # Indicate insufficient visibility
    
    lm = landmarks.landmark[idx]
    x_px = lm.x * img_width
    y_px = lm.y * img_height
    z_scaled = lm.z * img_width # Scale Z for consistency with X,Y units
    return np.array([x_px, y_px, z_scaled, lm.visibility])


def calculate_angle(p1_data, p2_data, p3_data):
    """
    Calculates the angle (in degrees) between three 3D points p1, p2, p3, with p2 as the vertex.
    Returns np.nan if any input data is None (due to low visibility).
    """
    if p1_data is None or p2_data is None or p3_data is None:
        return np.nan

    p1 = p1_data[:3] # Use X,Y,Z
    p2 = p2_data[:3]
    p3 = p3_data[:3]

    v1 = p1 - p2
    v2 = p3 - p2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0 # Or np.nan, depending on how you want to handle degenerate cases
    
    cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def calculate_all_features(current_landmarks_obj, prev_landmarks_obj, prev_prev_landmarks_obj, img_width, img_height):
    """
    Calculates a comprehensive set of features for a single frame,
    handling low visibility by returning NaN.
    """
    features = {}
    current_landmarks = current_landmarks_obj.landmark # Access the list of landmarks

    # --- 1. Raw Normalized 3D Coordinates & Visibility ---
    for i, lm in enumerate(current_landmarks):
        features[f'LM{i}_norm_x'] = lm.x
        features[f'LM{i}_norm_y'] = lm.y
        features[f'LM{i}_norm_z'] = lm.z
        features[f'LM{i}_visibility'] = lm.visibility

    # --- 2. Joint Angles (calculated from pixel coords, only if visible) ---
    for angle_name, (p1_idx, p2_idx, p3_idx) in ANGLE_DEFINITIONS.items():
        p1_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, p1_idx, img_width, img_height)
        p2_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, p2_idx, img_width, img_height)
        p3_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, p3_idx, img_width, img_height)
        
        angle = calculate_angle(p1_data, p2_data, p3_data)
        features[angle_name] = angle

    # --- 3. Normalized Distances (3D Euclidean Distance, Normalized by Body Scale, only if visible) ---
    body_scale = 1.0 
    try:
        left_shoulder_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_SHOULDER"], img_width, img_height)
        right_shoulder_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["RIGHT_SHOULDER"], img_width, img_height)
        
        shoulder_width = 0.0
        if left_shoulder_data is not None and right_shoulder_data is not None:
            shoulder_width = np.linalg.norm(left_shoulder_data[:3] - right_shoulder_data[:3])

        left_hip_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_HIP"], img_width, img_height)
        right_hip_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["RIGHT_HIP"], img_width, img_height)
        
        hip_width = 0.0
        if left_hip_data is not None and right_hip_data is not None:
            hip_width = np.linalg.norm(left_hip_data[:3] - right_hip_data[:3])

        # Prioritize shoulder/hip width as body scale, fallback to image height
        if shoulder_width > 0.05 * img_width: body_scale = shoulder_width
        elif hip_width > 0.05 * img_width: body_scale = hip_width
        else: body_scale = img_height / 3.0 # Fallback 

    except Exception: body_scale = img_height / 3.0
    
    if body_scale == 0: body_scale = 1.0 # Prevent division by zero


    wrist_dist = np.nan
    try:
        left_wrist_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_WRIST"], img_width, img_height)
        right_wrist_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["RIGHT_WRIST"], img_width, img_height)
        if left_wrist_data is not None and right_wrist_data is not None:
            wrist_dist = np.linalg.norm(left_wrist_data[:3] - right_wrist_data[:3]) / body_scale
    except Exception: pass
    features['DIST_LR_WRIST_NORM'] = wrist_dist

    ankle_dist = np.nan
    try:
        left_ankle_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_ANKLE"], img_width, img_height)
        right_ankle_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["RIGHT_ANKLE"], img_width, img_height)
        if left_ankle_data is not None and right_ankle_data is not None:
            ankle_dist = np.linalg.norm(left_ankle_data[:3] - right_ankle_data[:3]) / body_scale
    except Exception: pass
    features['DIST_LR_ANKLE_NORM'] = ankle_dist

    wrist_hip_dist = np.nan
    try:
        left_wrist_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_WRIST"], img_width, img_height)
        left_hip_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, LANDMARK_IDS["LEFT_HIP"], img_width, img_height)
        if left_wrist_data is not None and left_hip_data is not None:
            wrist_hip_dist = np.linalg.norm(left_wrist_data[:3] - left_hip_data[:3]) / body_scale
    except Exception: pass
    features['DIST_L_WRIST_HIP_NORM'] = wrist_hip_dist

    # --- 4. Relative Coordinates (X,Y,Z relative to mid-hip, using normalized MediaPipe coords) ---
    mid_hip_coords_norm = np.array([0.0, 0.0, 0.0]) # Default if hips not visible
    try:
        left_hip_norm_lm = current_landmarks[LANDMARK_IDS["LEFT_HIP"]]
        right_hip_norm_lm = current_landmarks[LANDMARK_IDS["RIGHT_HIP"]]
        if left_hip_norm_lm.visibility > MIN_LANDMARK_VISIBILITY and right_hip_norm_lm.visibility > MIN_LANDMARK_VISIBILITY:
            mid_hip_coords_norm = np.array([(left_hip_norm_lm.x + right_hip_norm_lm.x) / 2,
                                            (left_hip_norm_lm.y + right_hip_norm_lm.y) / 2,
                                            (left_hip_norm_lm.z + right_hip_norm_lm.z) / 2])
        else: # Fallback to image center if hips not visible
            mid_hip_coords_norm = np.array([0.5, 0.5, 0.0])
    except Exception:
        mid_hip_coords_norm = np.array([0.5, 0.5, 0.0])


    for i, lm in enumerate(current_landmarks):
        # Only calculate if landmark itself is visible
        if lm.visibility > MIN_LANDMARK_VISIBILITY:
            features[f'LM{i}_rel_x_norm'] = lm.x - mid_hip_coords_norm[0]
            features[f'LM{i}_rel_y_norm'] = lm.y - mid_hip_coords_norm[1]
            features[f'LM{i}_rel_z_norm'] = lm.z - mid_hip_coords_norm[2]
        else: # Mark as NaN if landmark not visible
            features[f'LM{i}_rel_x_norm'] = np.nan
            features[f'LM{i}_rel_y_norm'] = np.nan
            features[f'LM{i}_rel_z_norm'] = np.nan

    # --- 5. Dynamic Features (Velocities and Accelerations based on pixel coords) ---
    # Calculated only if all 3 frames (curr, prev, prev_prev) have sufficiently visible landmarks
    for i in range(len(current_landmarks)):
        curr_lm_data = get_landmark_coords_and_check_visibility(current_landmarks_obj, i, img_width, img_height)
        prev_lm_data = get_landmark_coords_and_check_visibility(prev_landmarks_obj, i, img_width, img_height) if prev_landmarks_obj else None
        prev_prev_lm_data = get_landmark_coords_and_check_visibility(prev_prev_landmarks_obj, i, img_width, img_height) if prev_prev_landmarks_obj else None

        if curr_lm_data is not None and prev_lm_data is not None and prev_prev_lm_data is not None:
            curr_lm_px = curr_lm_data[:3]
            prev_lm_px = prev_lm_data[:3]
            prev_prev_lm_px = prev_prev_lm_data[:3]

            velocity = curr_lm_px - prev_lm_px
            features[f'LM{i}_vx_px'] = velocity[0]
            features[f'LM{i}_vy_px'] = velocity[1]
            features[f'LM{i}_vz_px'] = velocity[2]

            acceleration = velocity - (prev_lm_px - prev_prev_lm_px)
            features[f'LM{i}_ax_px'] = acceleration[0]
            features[f'LM{i}_ay_px'] = acceleration[1]
            features[f'LM{i}_az_px'] = acceleration[2]
        else: # Mark as NaN if history or current landmark not visible
            features[f'LM{i}_vx_px'] = np.nan; features[f'LM{i}_vy_px'] = np.nan; features[f'LM{i}_vz_px'] = np.nan
            features[f'LM{i}_ax_px'] = np.nan; features[f'LM{i}_ay_px'] = np.nan; features[f'LM{i}_az_px'] = np.nan


    # --- 6. Contextual Feature (Variance Ratio - Torso Spread based on normalized coordinates) ---
    torso_var_ratio = np.nan
    try:
        torso_lm_indices = [LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["RIGHT_SHOULDER"], 
                            LANDMARK_IDS["LEFT_HIP"], LANDMARK_IDS["RIGHT_HIP"]]
        
        torso_x_coords = []
        torso_y_coords = []
        for idx in torso_lm_indices:
            lm = current_landmarks[idx]
            if lm.visibility > MIN_LANDMARK_VISIBILITY:
                torso_x_coords.append(lm.x)
                torso_y_coords.append(lm.y)
        
        if len(torso_x_coords) > 1 and len(torso_y_coords) > 1: # Need at least 2 points for variance
            var_x = np.var(torso_x_coords)
            var_y = np.var(torso_y_coords)
            torso_var_ratio = (var_x + 1e-6) / (var_y + 1e-6) # Add epsilon to prevent div by zero
    except Exception: pass
    features['TORSO_VAR_XY_RATIO'] = torso_var_ratio

    return features

# --- Drawing Function for Enhanced Skeleton ---
def draw_enhanced_skeleton(image, landmarks_obj, connections, min_draw_visibility=0.5):
    """
    Draws MediaPipe landmarks and connections on an image.
    Enhances drawing for major segments and highlights low-visibility landmarks.
    """
    annotated_image = image.copy()
    landmarks = landmarks_obj.landmark
    if not landmarks:
        return annotated_image # Return original if no landmarks

    # Define colors for drawing based on visibility
    HIGH_CONF_COLOR_POINT = (245, 117, 66) # Orange for highly visible points
    LOW_CONF_COLOR_POINT = (0, 0, 255)     # Red for low confidence points
    HIGH_CONF_COLOR_LINE = (245, 66, 230)  # Purple for high confidence connections
    LOW_CONF_COLOR_LINE = (0, 165, 255)    # Orange for low confidence connections

    DEFAULT_POINT_RADIUS = 3
    DEFAULT_LINE_THICKNESS = 2
    MAJOR_SEGMENT_THICKNESS = 5

    # Define major segments to draw thicker
    major_segments = [
        (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["RIGHT_SHOULDER"]), # Shoulders
        (LANDMARK_IDS["LEFT_HIP"], LANDMARK_IDS["RIGHT_HIP"]),         # Hips
        (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_HIP"]),     # Left Torso side
        (LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["RIGHT_HIP"]),   # Right Torso side
        (LANDMARK_IDS["LEFT_SHOULDER"], LANDMARK_IDS["LEFT_ELBOW"]),   # Left Upper Arm
        (LANDMARK_IDS["RIGHT_SHOULDER"], LANDMARK_IDS["RIGHT_ELBOW"]),  # Right Upper Arm
        (LANDMARK_IDS["LEFT_ELBOW"], LANDMARK_IDS["LEFT_WRIST"]),      # Left Forearm
        (LANDMARK_IDS["RIGHT_ELBOW"], LANDMARK_IDS["RIGHT_WRIST"]),    # Right Forearm
        (LANDMARK_IDS["LEFT_HIP"], LANDMARK_IDS["LEFT_KNEE"]),         # Left Thigh
        (LANDMARK_IDS["RIGHT_HIP"], LANDMARK_IDS["RIGHT_KNEE"]),       # Right Thigh
        (LANDMARK_IDS["LEFT_KNEE"], LANDMARK_IDS["LEFT_ANKLE"]),       # Left Shin
        (LANDMARK_IDS["RIGHT_KNEE"], LANDMARK_IDS["RIGHT_ANKLE"]),     # Right Shin
    ]

    # First pass: Draw connections (lines)
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_node = landmarks[start_idx]
            end_node = landmarks[end_idx]

            if start_node.visibility > min_draw_visibility and end_node.visibility > min_draw_visibility:
                color = HIGH_CONF_COLOR_LINE
            else:
                color = LOW_CONF_COLOR_LINE 

            p1 = (int(start_node.x * image.shape[1]), int(start_node.y * image.shape[0]))
            p2 = (int(end_node.x * image.shape[1]), int(end_node.y * image.shape[0]))

            if (start_idx, end_idx) in major_segments or (end_idx, start_idx) in major_segments:
                cv2.line(annotated_image, p1, p2, color, MAJOR_SEGMENT_THICKNESS)
            else:
                cv2.line(annotated_image, p1, p2, color, DEFAULT_LINE_THICKNESS)

    # Second pass: Draw landmarks (points)
    for i, landmark in enumerate(landmarks):
        if landmark.visibility > min_draw_visibility:
            color = HIGH_CONF_COLOR_POINT
            radius = DEFAULT_POINT_RADIUS
        else:
            color = LOW_CONF_COLOR_POINT
            radius = max(1, DEFAULT_POINT_RADIUS - 1)

        center = (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
        cv2.circle(annotated_image, center, radius, color, -1)

    return annotated_image

# --- Main Processing Loop ---

def process_image_sequences(raw_root, processed_root):
    """
    Processes image sequences from raw_root, extracts MediaPipe landmarks,
    calculates comprehensive features (with visibility checks), and saves
    annotated images (original + skeleton, highlighting visibility).
    
    Args:
        raw_root (str): The root directory containing your 'train', 'test', 'valid' splits,
                        where each split folder contains video_clip_X folders with
                        'frame_0000x.jpg' images (output from Frame_Renaming_Staged.py).
        processed_root (str): The root directory where processed features (CSVs)
                              and annotated images will be saved.
    """
    os.makedirs(processed_root, exist_ok=True)

    print(f"Initializing MediaPipe Pose model...")
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, static_image_mode=False) as pose:
        print("MediaPipe Pose model initialized successfully.")
        
        for split in ['train', 'test', 'valid']:
            split_raw_path = os.path.join(raw_root, split)
            split_processed_path = os.path.join(processed_root, split)
            os.makedirs(split_processed_path, exist_ok=True)

            if not os.path.exists(split_raw_path):
                print(f"Warning: Raw path for {split} split '{split_raw_path}' not found. Skipping {split} split.")
                continue

            print(f"--- Processing {split} split ---")
            
            video_clip_dirs = sorted([d for d in os.listdir(split_raw_path) if os.path.isdir(os.path.join(split_raw_path, d))])
            
            if not video_clip_dirs:
                print(f"No video clip directories found in {split_raw_path}. Skipping.")
                continue

            for clip_name in video_clip_dirs:
                clip_raw_path = os.path.join(split_raw_path, clip_name) 
                
                clip_annotated_output_dir = os.path.join(split_processed_path, f"{clip_name}_annotated_images")
                clip_features_output_csv = os.path.join(split_processed_path, f"{clip_name}_features.csv")
                
                os.makedirs(clip_annotated_output_dir, exist_ok=True)

                print(f"  Processing clip: {clip_name}")
                
                image_files = sorted([f for f in os.listdir(clip_raw_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
                
                if not image_files:
                    print(f"    No image files found in {clip_raw_path}. Skipping clip.")
                    continue

                all_frame_features_data = []
                landmark_history_buffer = deque(maxlen=2) 

                for i, img_filename in enumerate(image_files):
                    frame_path = os.path.join(clip_raw_path, img_filename)
                    frame_bgr = cv2.imread(frame_path)

                    if frame_bgr is None:
                        print(f"    Could not read image: {frame_path}. Skipping.")
                        continue

                    h, w, _ = frame_bgr.shape

                    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False

                    results = pose.process(rgb_frame)

                    current_frame_data = {
                        'clip_id': clip_name,
                        'frame_index': i,
                        'original_image_filename': img_filename, # This will be frame_0000x.jpg here!
                        'annotated_image_path': '' 
                    }

                    annotated_image_bgr = frame_bgr.copy() 
                    
                    if results.pose_landmarks:
                        annotated_image_bgr = draw_enhanced_skeleton(
                            annotated_image_bgr,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            min_draw_visibility=MIN_LANDMARK_VISIBILITY
                        )
                            
                        # --- Calculate Features ---
                        prev_lm_obj = landmark_history_buffer[0] if len(landmark_history_buffer) >= 1 else None
                        prev_prev_lm_obj = landmark_history_buffer[1] if len(landmark_history_buffer) >= 2 else None
                        
                        features = calculate_all_features(results.pose_landmarks, prev_lm_obj, prev_prev_lm_obj, w, h)
                        current_frame_data.update(features)

                        landmark_history_buffer.appendleft(results.pose_landmarks)
                    else:
                        print(f"    No pose landmarks detected for {img_filename}. Filling with NaNs for features.")
                        # Fill all feature columns with NaNs if no pose is detected to maintain DataFrame consistency
                        dummy_features = {}
                        for j in range(33):
                            for coord in ['x','y','z','visibility']: dummy_features[f'LM{j}_norm_{coord}'] = np.nan
                        for name in ANGLE_DEFINITIONS.keys(): dummy_features[name] = np.nan
                        dummy_features.update({'DIST_LR_WRIST_NORM': np.nan, 'DIST_LR_ANKLE_NORM': np.nan, 'DIST_L_WRIST_HIP_NORM': np.nan})
                        for j in range(33):
                            for coord in ['rel_x_norm','rel_y_norm','rel_z_norm']: dummy_features[f'LM{j}_rel_{coord}'] = np.nan
                        for j in range(33):
                            for coord in ['vx_px','vy_px','vz_px','ax_px','ay_px','az_px']: dummy_features[f'LM{j}_{coord}'] = np.nan
                        dummy_features.update({'TORSO_VAR_XY_RATIO': np.nan})
                        
                        current_frame_data.update(dummy_features)


                    # Save the annotated image
                    annotated_filename = f"{os.path.splitext(img_filename)[0]}_annotated.jpg"
                    annotated_output_path = os.path.join(clip_annotated_output_dir, annotated_filename)
                    cv2.imwrite(annotated_output_path, annotated_image_bgr)
                    current_frame_data['annotated_image_path'] = os.path.relpath(annotated_output_path, start=PROCESSED_DATA_ROOT)


                    all_frame_features_data.append(current_frame_data)

                # Save all features for the current clip to a CSV
                if all_frame_features_data:
                    df_clip = pd.DataFrame(all_frame_features_data)
                    df_clip.to_csv(clip_features_output_csv, index=False)
                    print(f"  Saved features for {clip_name} to {clip_features_output_csv}")
                else:
                    print(f"  No frame data or features to save for {clip_name}. This clip may have no valid images.")

    print("\n--- All image sequences processed! ---")
    print(f"Processed data saved to: {PROCESSED_DATA_ROOT}")


# --- Run the script ---
if __name__ == "__main__":
    process_image_sequences(RAW_IMAGE_SEQUENCES_ROOT, PROCESSED_DATA_ROOT)

