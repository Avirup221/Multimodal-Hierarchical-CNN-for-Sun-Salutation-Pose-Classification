import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from tqdm import tqdm # For progress bar

# Import your Multimodal QuadtreeCNN model definition and DataLoader
from models import QuadtreeCNN, get_model, StandardResNetCNN 
from dataloader import YogaPoseFrameDataset, IMAGE_SIZE 

# --- Configuration ---
# IMPORTANT: Set this to the mode of the model you want to visualize!
# You will run this script multiple times, changing this mode each time:
# - 'fusion'
# - 'image_only'
# - 'standard_resnet_only'
# (Do NOT run for 'numerical_only' as Grad-CAM is not applicable for it)
VISUALIZATION_MODE = 'standard_resnet_only' # <<< CHANGE THIS FOR EACH EXPERIMENT

# Path to your trained model weights (automatically determined by VISUALIZATION_MODE)
MODEL_PATH = f'multimodal_quadtree_cnn_pose_model_{VISUALIZATION_MODE}.pth' 

# Path to your flat dataset root (where 'train', 'test', 'valid' folders are)
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'
CLASS_FEATURE_MEANS_FILE = os.path.join(STILL_IMAGE_DATASET_ROOT, 'class_feature_means.json')

# Output directory for saved Grad-CAM images. A subfolder will be created for each VISUALIZATION_MODE.
OUTPUT_GRAD_CAM_DIR = os.path.join(STILL_IMAGE_DATASET_ROOT, f'grad_cam_visualizations_{VISUALIZATION_MODE}')

# Model and Image Constants (MUST match training config)
NUM_NUMERICAL_FEATURES = 47 

# Set device for inference
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for Batch Grad-CAM (Mode: {VISUALIZATION_MODE})")

# --- Image Transformation for Inference (no augmentation) ---
image_transform_inference = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Feature Extraction Helper Functions (Copied for self-containment) ---
# These are identical to those in 1_prepare_still_image_dataset.py
import mediapipe as mp
mp_pose = mp.solutions.pose
POSE = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

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
    angle_deg = np.degrees(np.arccos(angle_rad))
    
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

def preprocess_image_for_cam(image_path):
    original_image_bgr = cv2.imread(image_path)
    if original_image_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    original_image_rgb = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(original_image_rgb)
    
    input_tensor = image_transform_inference(image_pil).unsqueeze(0).to(device)
    
    return original_image_bgr, input_tensor

def generate_grad_cam_heatmap(model, image_input_tensor, numerical_input_tensor, target_class=None, visualization_mode='fusion'):
    # Grad-CAM is not applicable for models that rely solely on numerical features.
    if visualization_mode == 'numerical_only':
        # print("Skipping Grad-CAM for 'numerical_only' mode as it's not applicable to non-image inputs.")
        return None

    model.eval()
    
    # Ensure gradients are enabled for the image input for Grad-CAM
    # Clone the tensor to avoid modifying the original input_tensor if it's used elsewhere
    image_input_tensor_grad = image_input_tensor.clone().requires_grad_(True) 

    # Determine the target layer for Grad-CAM based on the model type
    # Both QuadtreeCNN and StandardResNetCNN use `base_cnn.layer4` as their last conv layer before pooling
    if isinstance(model, QuadtreeCNN) or isinstance(model, StandardResNetCNN):
        target_layer = model.base_cnn.layer4
    else:
        raise TypeError("Unsupported model type for Grad-CAM. Must be QuadtreeCNN or StandardResNetCNN.")

    # Register hooks directly on the target layer
    # These hooks will populate model.activations and model.gradients during the forward/backward pass
    hook_handle_activation = target_layer.register_forward_hook(model.save_activation_hook)
    hook_handle_gradient = target_layer.register_full_backward_hook(model.save_gradient_hook)

    # Perform the full forward pass to get logits
    # Pass the correct inputs based on the model's actual mode
    if visualization_mode in ['image_only', 'standard_resnet_only']: 
        # For image_only and standard_resnet_only, numerical_input_tensor is ignored by the model's forward
        logits = model(image_input_tensor_grad, torch.empty(1, NUM_NUMERICAL_FEATURES).to(device)) 
    elif visualization_mode == 'fusion':
        logits = model(image_input_tensor_grad, numerical_input_tensor)
    else:
        # This case should ideally be caught by the initial check, but for safety
        print(f"Warning: Unexpected visualization_mode '{visualization_mode}' for Grad-CAM. Returning None.")
        return None

    # Get the predicted class if target_class is not specified
    if target_class is None: 
        target_class = torch.argmax(logits).item()

    # Zero gradients before backward pass
    model.zero_grad()
    
    # Create a one-hot vector for the target class
    one_hot_output = torch.zeros_like(logits).to(device)
    one_hot_output[0][target_class] = 1 

    # Perform backward pass to get gradients
    # This will trigger the backward hook and populate model.gradients
    logits.backward(gradient=one_hot_output, retain_graph=True) 

    # Get the captured gradients and activations
    gradients_tensor = model.gradients 
    activations_tensor = model.activations 
    
    # Remove hooks to prevent interference with other operations
    hook_handle_activation.remove()
    hook_handle_gradient.remove()

    # Clear model's internal attributes for next call (good practice)
    model.gradients = None
    model.activations = None

    # Safeguard check: If for some reason gradients or activations are None, return None
    if gradients_tensor is None or activations_tensor is None:
        print("Warning: Gradients or Activations were None after backward pass. Grad-CAM might not be applicable or hooks failed.")
        return None

    # Global average pooling of gradients
    pooled_gradients = torch.mean(gradients_tensor, dim=[2, 3])
    
    # Weight the channels by their corresponding gradients
    activations_tensor_detached = activations_tensor.detach()
    for i in range(activations_tensor_detached.shape[0]): 
        for j in range(activations_tensor_detached.shape[1]): 
            activations_tensor_detached[i, j, :, :] *= pooled_gradients[i, j]

    # Sum across channels to get the heatmap
    heatmap = torch.sum(activations_tensor_detached, dim=1).squeeze() 
    heatmap = nn.functional.relu(heatmap)

    # Normalize heatmap to 0-1 range and convert to NumPy
    heatmap = heatmap.cpu().numpy()
    heatmap = np.maximum(heatmap, 0) 
    if np.max(heatmap) == 0: 
        heatmap = np.zeros_like(heatmap)
    else:
        heatmap /= np.max(heatmap)

    return heatmap

def visualize_cam(original_image_bgr, heatmap, alpha=0.4):
    if heatmap is None: # Handle case where heatmap was not generated
        return original_image_bgr # Return original image if no heatmap

    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (original_image_bgr.shape[1], original_image_bgr.shape[0]))
    
    # Apply colormap to heatmap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    superimposed_img = heatmap * alpha + original_image_bgr * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img


# --- Main Execution ---
if __name__ == '__main__':
    # Load class names
    class_names = load_class_names(CLASS_FEATURE_MEANS_FILE)
    num_classes = len(class_names)
    print(f"Loaded class names: {class_names}")

    # Initialize model with the specified VISUALIZATION_MODE
    model = get_model(num_classes=num_classes, numerical_feature_dim=NUM_NUMERICAL_FEATURES, mode=VISUALIZATION_MODE, device=device).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"Model weights loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model weights from {MODEL_PATH}: {e}")
            print("Please ensure the model architecture in models.py matches the saved weights.")
            exit()
    else:
        print(f"Error: Model weights not found at {MODEL_PATH}. Please ensure training completed successfully.")
        exit()

    # --- Prepare DataLoader for the training set ---
    train_data_path = os.path.join(STILL_IMAGE_DATASET_ROOT, 'train')
    if not os.path.exists(train_data_path):
        print(f"Error: Training data path not found: {train_data_path}. Please ensure 1_prepare_still_image_dataset.py ran correctly.")
        exit()

    print(f"\nLoading training dataset from: {train_data_path} for Grad-CAM analysis...")
    # Use batch_size=1 for Grad-CAM as it's typically computed per image
    # is_train=False to avoid random augmentations during visualization
    train_dataset = YogaPoseFrameDataset(train_data_path, IMAGE_SIZE, is_train=False) 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
    
    print(f"Total images in training dataset: {len(train_dataset)}")

    # Create output directory for Grad-CAM images
    os.makedirs(OUTPUT_GRAD_CAM_DIR, exist_ok=True)
    print(f"Saving Grad-CAM visualizations to: {OUTPUT_GRAD_CAM_DIR}")

    # Iterate through the training dataset
    for i, (image_tensor, numerical_features_tensor, label_tensor) in enumerate(tqdm(train_loader, desc="Generating Grad-CAMs")):
        # Get original image path from dataset to load BGR version
        original_image_path = train_dataset.image_paths[i]
        original_image_bgr = cv2.imread(original_image_path)
        
        if original_image_bgr is None:
            print(f"Warning: Could not read original image {original_image_path}. Skipping Grad-CAM for this image.")
            continue

        # Move tensors to device
        image_tensor = image_tensor.to(device)
        numerical_features_tensor = numerical_features_tensor.to(device)
        
        # Get model prediction for the image
        model.eval()
        with torch.no_grad(): # Use no_grad for the initial prediction to avoid unnecessary graph building
            outputs_initial = model(image_tensor, numerical_features_tensor) 
            probabilities_initial = torch.softmax(outputs_initial, dim=1)
            confidence_initial, predicted_idx_initial = torch.max(probabilities_initial, 1)
            predicted_label_initial = class_names[predicted_idx_initial.item()]
            confidence_score_initial = confidence_initial.item() # Convert to Python float

        # Generate Grad-CAM Heatmap
        # Pass the visualization_mode to the heatmap generation function
        heatmap = generate_grad_cam_heatmap(model, image_tensor, numerical_features_tensor, 
                                            target_class=predicted_idx_initial.item(), 
                                            visualization_mode=VISUALIZATION_MODE)

        # Visualize and Save
        # Only visualize if a heatmap was generated (i.e., not for 'numerical_only' mode)
        if heatmap is not None:
            cam_image = visualize_cam(original_image_bgr, heatmap)

            # Create class-specific subdirectory for output
            class_output_dir = os.path.join(OUTPUT_GRAD_CAM_DIR, class_names[label_tensor.item()])
            os.makedirs(class_output_dir, exist_ok=True)

            # Save the Grad-CAM image
            base_filename = os.path.splitext(os.path.basename(original_image_path))[0]
            output_cam_path = os.path.join(class_output_dir, f"{base_filename}_pred_{predicted_label_initial}_cam.jpg")
            cv2.imwrite(output_cam_path, cam_image)
        else:
            # For numerical_only mode, save the original image with prediction text
            # print(f"Grad-CAM not generated for {VISUALIZATION_MODE} mode for {os.path.basename(original_image_path)}.") # Uncomment for more verbose output
            class_output_dir = os.path.join(OUTPUT_GRAD_CAM_DIR, class_names[label_tensor.item()])
            os.makedirs(class_output_dir, exist_ok=True)
            output_original_path = os.path.join(class_output_dir, f"pred_{predicted_label_initial}_{os.path.splitext(os.path.basename(original_image_path))[0]}_{VISUALIZATION_MODE}.jpg")
            
            temp_img = original_image_bgr.copy()
            # Ensure confidence_initial is converted to float for text rendering
            cv2.putText(temp_img, f"Predicted: {predicted_label_initial} (Confidence: {confidence_initial.item():.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(output_original_path, temp_img)
            # print(f"Original image with prediction saved to: {output_original_path}") # Uncomment for more verbose output

    print(f"\nBatch Grad-CAM analysis complete. Visualizations saved to: {OUTPUT_GRAD_CAM_DIR}")
    print("You can now inspect the generated images in the output directory.")

