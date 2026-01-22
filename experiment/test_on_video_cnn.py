import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import mediapipe as mp
import json
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Import your Multimodal QuadtreeCNN model definition
from models_cnn import QuadtreeCNN, get_model 

# --- Configuration ---
# Path to your class-wise feature means JSON file (for numerical feature imputation during training)
# We will load class names from this file.
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'
CLASS_FEATURE_MEANS_FILE = os.path.join(STILL_IMAGE_DATASET_ROOT, 'class_feature_means.json')

# Model and Image Constants (MUST match training config)
IMAGE_SIZE = (224, 224) 
NUM_NUMERICAL_FEATURES = 47 

# MediaPipe Pose configuration (MUST match data preparation config)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
POSE = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Device for inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# --- Ablation Study Mode ---
# IMPORTANT: Change this variable to match the mode of the model you want to use for inference!
INFERENCE_MODE = 'fusion' # <<< CHANGE THIS FOR EACH EXPERIMENT

# Path to your trained model weights (will now depend on the mode)
MODEL_PATH = f'multimodal_quadtree_cnn_pose_model_{INFERENCE_MODE}.pth' 

# --- Feature Extraction Helper Functions (Copied from 1_prepare_still_image_dataset.py) ---
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2) 
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_landmark_visibility(landmarks):
    visibility = []
    for lm in landmarks.landmark:
        visibility.append(lm.visibility)
    return np.array(visibility)

def calculate_torso_angle_vertical(landmarks):
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

SELECTED_FEATURE_COLUMNS = []
for i in range(33): 
    SELECTED_FEATURE_COLUMNS.append(f'LM{i}_visibility')
SELECTED_FEATURE_COLUMNS.extend([
    'LEFT_ELBOW_ANGLE', 'RIGHT_ELBOW_ANGLE', 'LEFT_SHOULDER_ANGLE', 'RIGHT_SHOULDER_ANGLE',
    'LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 'LEFT_HIP_ANGLE', 'RIGHT_HIP_ANGLE',
    'TORSO_VERTICAL_ANGLE', 'TORSO_HORIZONTAL_ALIGNMENT'
])
SELECTED_FEATURE_COLUMNS.extend([
    'DIST_LR_WRIST_NORM', 'DIST_LR_ANKLE_NORM', 'DIST_L_WRIST_HIP_NORM'
])
SELECTED_FEATURE_COLUMNS.append('TORSO_VAR_XY_RATIO')
assert len(SELECTED_FEATURE_COLUMNS) == 47, f"Expected 47 features, but got {len(SELECTED_FEATURE_COLUMNS)}"


def extract_and_process_features(image_rgb, img_w, img_h):
    results = POSE.process(image_rgb)
    current_frame_features_dict = {}

    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        
        lm_visibility = get_landmark_visibility(landmarks)
        for j in range(33):
            current_frame_features_dict[f'LM{j}_visibility'] = lm_visibility[j] 
        
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

        try:
            torso_lms_x = [landmarks.landmark[11].x, landmarks.landmark[12].x, landmarks.landmark[23].x, landmarks.landmark[24].x]
            torso_lms_y = [landmarks.landmark[11].y, landmarks.landmark[12].y, landmarks.landmark[23].y, landmarks.landmark[24].y]
            
            visible_torso_lms_x = [x for k, x in enumerate(torso_lms_x) if landmarks.landmark[[11,12,23,24][k]].visibility > 0.65]
            visible_torso_lms_y = [y for k, y in enumerate(torso_lms_y) if landmarks.landmark[[11,12,23,24][k]].visibility > 0.65]

            if len(visible_torso_lms_x) >= 2 and len(visible_torso_lms_y) >= 2: 
                var_x = np.var(visible_torso_lms_x)
                var_y = np.var(visible_torso_lms_y)
                current_frame_features_dict['TORSO_VAR_XY_RATIO'] = var_x / var_y if var_y != 0 else np.nan 
            else:
                current_frame_features_dict['TORSO_VAR_XY_RATIO'] = np.nan
        except:
            current_frame_features_dict['TORSO_VAR_XY_RATIO'] = np.nan

    else: # No pose landmarks detected - fill all selected features with NaN
        for col in SELECTED_FEATURE_COLUMNS:
            current_frame_features_dict[col] = np.nan 
        for j in range(33):
            current_frame_features_dict[f'LM{j}_visibility'] = 0.0 # Explicitly zero visibility if no pose

    numerical_features_array = np.array([current_frame_features_dict.get(col, np.nan) for col in SELECTED_FEATURE_COLUMNS], dtype=np.float32)
    return numerical_features_array


def load_class_names(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Class feature means file not found: {file_path}. Please ensure 1_prepare_still_image_dataset.py ran correctly.")
    with open(file_path, 'r') as f:
        data = json.load(f)
        return sorted(list(data.keys())) 

def run_video_inference(video_path, model, class_names, output_video_path=None):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Processing video: {os.path.basename(video_path)} ({int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames, {fps:.2f} FPS)")

    image_transform_inference = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    out = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            print(f"Warning: Could not open video writer for {output_video_path}. Will not save output video.")
            out = None

    model.eval()

    frame_idx = 0
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1

            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = image_transform_inference(image_pil).unsqueeze(0).to(device) 

            img_h, img_w, _ = frame.shape
            numerical_features_np = extract_and_process_features(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), img_w, img_h)
            numerical_features_tensor = torch.tensor(numerical_features_np, dtype=torch.float32).unsqueeze(0).to(device) 

            # Impute NaNs for inference: Use 0.0 as fallback when true class mean is unknown
            numerical_features_tensor[numerical_features_tensor.isnan()] = 0.0 

            # --- Conditional Model Input based on INFERENCE_MODE ---
            if INFERENCE_MODE == 'image_only':
                outputs = model(image_tensor, torch.empty(1, NUM_NUMERICAL_FEATURES).to(device)) # Pass dummy numerical input
            elif INFERENCE_MODE == 'numerical_only':
                outputs = model(torch.empty(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(device), numerical_features_tensor) # Pass dummy image input
            elif INFERENCE_MODE == 'fusion':
                outputs = model(image_tensor, numerical_features_tensor)
            else:
                raise ValueError(f"Invalid INFERENCE_MODE: {INFERENCE_MODE}")
            # --- END Conditional Model Input ---

            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_label = class_names[predicted_idx.item()]
            confidence_score = confidence.item()

            frame_rgb_mp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = POSE.process(frame_rgb_mp) 

            if mp_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    mp_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

            text_to_display = f"Pose: {predicted_label} ({confidence_score:.2f})"
            cv2.putText(frame, text_to_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Yoga Pose Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if out:
                out.write(frame)

    cap.release()
    if out:
        out.release()
        print(f"Output video saved to {output_video_path}")
    cv2.destroyAllWindows()
    print(f"Finished processing {frame_idx} frames from {video_path}.")


# --- Main Execution ---
if __name__ == '__main__':
    class_names = load_class_names(CLASS_FEATURE_MEANS_FILE)
    num_classes = len(class_names)
    print(f"Loaded class names: {class_names}")

    # Initialize model with the specified INFERENCE_MODE
    model = get_model(num_classes=num_classes, numerical_feature_dim=NUM_NUMERICAL_FEATURES, mode=INFERENCE_MODE).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Model weights loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Please ensure the model architecture in models.py matches the saved weights.")
            exit()
    else:
        print(f"Error: Model weights not found at {MODEL_PATH}. Please ensure training completed successfully.")
        exit()

    video_to_test_path = r'E:/User/my work/Summer project/Code/Model_Testing/testing.mp4' 
    output_video_save_path = r'E:/User/my work/Summer project/Code/Model_Testing/output_testing_video_predictions.mp4' 

    print(f"\nStarting live prediction for video: {video_to_test_path} (Mode: {INFERENCE_MODE})")
    print("Press 'q' to quit the video playback window.")
    run_video_inference(video_to_test_path, model, class_names, output_video_save_path)

