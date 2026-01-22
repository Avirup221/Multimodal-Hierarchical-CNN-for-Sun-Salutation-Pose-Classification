import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# --- Paste the HierarchicalQuadtreeCNN class definition ---
class HierarchicalQuadtreeCNN(nn.Module):
    def __init__(self, num_classes, numerical_feature_dim=47, dropout_rate=0.5):
        super().__init__()
        base_cnn = models.resnet18(weights=None)
        self.features_extractor = nn.Sequential(
            base_cnn.conv1, base_cnn.bn1, base_cnn.relu, base_cnn.maxpool,
            base_cnn.layer1, base_cnn.layer2
        )
        # ... (rest of the class definition is the same)
        base_feature_channels = 128
        self.global_processor = nn.Sequential(
            base_cnn.layer3, base_cnn.layer4, base_cnn.avgpool
        )
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1))
        )
        self.sub_quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1))
        )
        total_image_feature_dim = 512 + (4 * 128) + (16 * 64)
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate)
        )
        combined_feature_dim = total_image_feature_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 1024),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
    
    # Custom forward method to extract intermediate feature maps for visualization
    def forward_for_visualization(self, image_input):
        # Level 0
        base_features = self.features_extractor(image_input)
        
        # Level 1
        h, w = base_features.shape[2], base_features.shape[3]
        mid_h, mid_w = h // 2, w // 2
        q1 = base_features[:, :, :mid_h, :mid_w] # Top-left quadrant
        
        # Level 2
        qh, qw = q1.shape[2], q1.shape[3]
        mid_qh, mid_qw = qh // 2, qw // 2
        sq1_1 = q1[:, :, :mid_qh, :mid_qw] # Top-left sub-quadrant of Q1

        return base_features, q1, sq1_1

# --- Main Visualization Script ---
if __name__ == '__main__':
    # --- CONFIGURATION ---
    MODEL_WEIGHTS_PATH = 'multimodal_hierarchical_quadtree_pose_model.pth'
    IMAGE_PATH =r"E:\User\my work\Summer project\Code\data_preprocessing\RenamedDataset\train\video_clip_005\frame_00052.jpg"
    NUM_CLASSES = 8 # IMPORTANT: Change this to the correct number of classes
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    model = HierarchicalQuadtreeCNN(num_classes=NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()
    model.eval()

    # --- Load and Preprocess Image ---
    try:
        image = Image.open(IMAGE_PATH).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at '{IMAGE_PATH}'. Please check the path.")
        exit()
        
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    # --- Get Feature Maps from Each Level ---
    with torch.no_grad():
        base_map, quadrant_map, sub_quadrant_map = model.forward_for_visualization(image_tensor)

    # Function to convert a feature map tensor to a displayable heatmap
    def to_heatmap(feature_map):
        return np.mean(feature_map.squeeze(0).cpu().numpy(), axis=0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Hierarchical Breakdown of Feature Maps', fontsize=20)

    # 1. Original Image
    axes[0].imshow(image.resize((224, 224)))
    axes[0].set_title('Original Image (224x224)')
    axes[0].axis('off')

    # 2. Level 0 - Base Feature Map
    heatmap_base = to_heatmap(base_map)
    axes[1].imshow(heatmap_base, cmap='viridis')
    axes[1].set_title('Level 0: Base Map (28x28)')
    # Draw quadrant lines
    h, w = heatmap_base.shape
    axes[1].axvline(x=(w/2 - 0.5), color='r', linestyle='--')
    axes[1].axhline(y=(h/2 - 0.5), color='r', linestyle='--')
    axes[1].axis('off')

    # 3. Level 1 - Quadrant Feature Map
    heatmap_quad = to_heatmap(quadrant_map)
    axes[2].imshow(heatmap_quad, cmap='viridis')
    axes[2].set_title('Level 1: Quadrant (14x14)')
    axes[2].axis('off')

    # 4. Level 2 - Sub-Quadrant Feature Map
    heatmap_sub_quad = to_heatmap(sub_quadrant_map)
    axes[3].imshow(heatmap_sub_quad, cmap='viridis')
    axes[3].set_title('Level 2: Sub-Quadrant (7x7)')
    axes[3].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()