import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np

# Import necessary modules from your project files
from models_cnn import get_model # To load the Multimodal QuadtreeCNN architecture
from dataloader_cnn import YogaPoseFrameDataset, IMAGE_SIZE 

# --- Configuration Paths (MUST Match Training Configuration) ---
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'

# --- Ablation Study Mode ---
# IMPORTANT: Change this variable to match the mode of the model you want to evaluate!
EVALUATION_MODE = 'numerical_only' # <<< CHANGE THIS TO MATCH THE TRAINED MODEL

# Path to the trained model weights (will now depend on the mode)
MODEL_PATH = f'multimodal_quadtree_cnn_pose_model_{EVALUATION_MODE}.pth' 

BATCH_SIZE = 16 
RANDOM_SEED = 42 

# Ensure reproducibility (same as training)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device for evaluation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device for evaluation: {device} for mode: {EVALUATION_MODE}")

if __name__ == '__main__':
    # --- 1. Load Test Data ---
    test_data_path = os.path.join(STILL_IMAGE_DATASET_ROOT, 'test')
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data path not found: {test_data_path}. Please ensure your data pipeline ran correctly.")
        exit()

    print(f"Loading test data from: {test_data_path}")
    test_dataset = YogaPoseFrameDataset(test_data_path, IMAGE_SIZE, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)

    class_names = test_dataset.idx_to_class
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}, Class names: {class_names}")
    print(f"Total test samples loaded: {len(test_dataset)}")

    if len(test_dataset) == 0:
        print("No test samples found. Cannot perform evaluation.")
        exit()

    # --- 2. Load Model ---
    print(f"\nLoading model from {MODEL_PATH}...")
    # Pass the EVALUATION_MODE to the get_model function
    model = get_model(num_classes=num_classes, device=device, mode=EVALUATION_MODE, print_num_params=False)
    
    # Load the saved state_dict
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights from {MODEL_PATH}: {e}")
            print("Please ensure the model architecture in models.py matches the saved weights.")
            exit()
    else:
        print(f"Error: Model weights not found at {MODEL_PATH}. Please ensure training completed successfully.")
        exit()

    # Set model to evaluation mode
    model.eval()

    # --- 3. Evaluate Model on Test Set ---
    print("\nStarting evaluation on test set...")
    correct = 0
    total = 0
    
    test_pbar = tqdm(test_loader, desc=f"Evaluating on Test Set ({EVALUATION_MODE})", leave=False)

    with torch.no_grad(): 
        for images, numerical_features, labels in test_pbar:
            images = images.to(device)
            numerical_features = numerical_features.to(device) 
            labels = labels.to(device)

            # Pass both inputs to the model, model's forward handles the mode
            outputs = model(images, numerical_features)
            _, predicted = torch.max(outputs.data, 1) 

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = (correct / total) * 100
    print(f"\n--- Test Set Evaluation Complete ({EVALUATION_MODE}) ---")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

