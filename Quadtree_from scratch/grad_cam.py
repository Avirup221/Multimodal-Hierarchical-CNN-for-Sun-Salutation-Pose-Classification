import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.models as models

# --- 1. QuadtreeCNN Class with Hook Methods Added ---
# This is the model from your models.py file, modified to support your preferred Grad-CAM logic.
class QuadtreeCNN(nn.Module):
    def __init__(self, num_classes, cnn_feature_dim=512, numerical_feature_dim=47, dropout_rate=0.5):
        super(QuadtreeCNN, self).__init__()
        
        # --- Attributes for Grad-CAM ---
        self.gradients = None
        self.activations = None

        # --- Model Architecture ---
        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features_extractor = nn.Sequential(
            self.base_cnn.conv1, self.base_cnn.bn1, self.base_cnn.relu, self.base_cnn.maxpool,
            self.base_cnn.layer1, self.base_cnn.layer2, self.base_cnn.layer3
        )
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(256, cnn_feature_dim // 4, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_processor = nn.Sequential(self.base_cnn.layer4, self.base_cnn.avgpool)
        self.image_feature_dim = 512 + ((cnn_feature_dim // 4) * 3 * 3 * 4)
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(numerical_feature_dim * 2, cnn_feature_dim // 2)
        )
        self.numerical_output_dim = cnn_feature_dim // 2
        self.combined_feature_dim = self.image_feature_dim + self.numerical_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.combined_feature_dim // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(self.combined_feature_dim // 2, num_classes)
        )

    # --- Hook Methods for Grad-CAM ---
    def save_activation_hook(self, module, input, output):
        self.activations = output

    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, image_input, numerical_input):
        base_features = self.features_extractor(image_input) 
        h, w = base_features.shape[2], base_features.shape[3]
        q1 = base_features[:, :, :h//2, :w//2]; q2 = base_features[:, :, :h//2, w//2:]
        q3 = base_features[:, :, h//2:, :w//2]; q4 = base_features[:, :, h//2:, w//2:]
        q1_features = self.quadrant_processor(q1).flatten(1); q2_features = self.quadrant_processor(q2).flatten(1)
        q3_features = self.quadrant_processor(q3).flatten(1); q4_features = self.quadrant_processor(q4).flatten(1)
        global_features = self.global_processor(base_features).flatten(1) 
        image_features = torch.cat((global_features, q1_features, q2_features, q3_features, q4_features), dim=1)
        numerical_features = self.numerical_mlp(numerical_input)
        combined_features = torch.cat((image_features, numerical_features), dim=1)
        logits = self.classifier(combined_features)
        return logits


# --- 2. Grad-CAM Logic ---
def generate_grad_cam_heatmap(model, image_input_tensor, numerical_input_tensor, target_layer, target_class=None):
    model.eval()
    hook_handle_activation = target_layer.register_forward_hook(model.save_activation_hook)
    hook_handle_gradient = target_layer.register_full_backward_hook(model.save_gradient_hook)
    logits = model(image_input_tensor, numerical_input_tensor)
    if target_class is None: target_class = torch.argmax(logits).item()
    model.zero_grad()
    one_hot_output = torch.zeros_like(logits)
    one_hot_output[0][target_class] = 1
    logits.backward(gradient=one_hot_output, retain_graph=True)
    gradients = model.gradients; activations = model.activations
    hook_handle_activation.remove(); hook_handle_gradient.remove()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]): activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = nn.functional.relu(heatmap)
    heatmap /= torch.max(heatmap)
    return heatmap.detach().cpu().numpy(), target_class

def visualize_cam(original_image, heatmap, alpha=0.5):
    img_np = np.array(original_image)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap_colored * alpha + img_np * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)
    return superimposed_img


# --- 3. Main Execution for a Single Image ---
if __name__ == '__main__':
    # --- CONFIGURATION: UPDATE THESE THREE VARIABLES ---
    MODEL_PATH = 'quadtree_pose_model.pth'
    IMAGE_PATH = 'frame_00146.jpg'
    NUM_CLASSES = 8 # IMPORTANT: Must match the trained model

    # --- SETUP ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuadtreeCNN(num_classes=NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()

    # --- LOAD AND PREPROCESS IMAGE ---
    image_pil = Image.open(IMAGE_PATH).convert('RGB').resize((224, 224))
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)
    dummy_numerical_features = torch.randn(1, 47).to(DEVICE)

    # --- GENERATE GRAD-CAM ---
    # Target the last convolutional layer in the ResNet backbone
    target_layer_for_cam = model.base_cnn.layer4
    
    heatmap, predicted_idx = generate_grad_cam_heatmap(
        model, 
        input_tensor, 
        dummy_numerical_features, 
        target_layer=target_layer_for_cam
    )
    print(f"Predicted class index: {predicted_idx}")
    
    # --- VISUALIZATION ---
    cam_image = visualize_cam(image_pil, heatmap)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('QuadtreeCNN Grad-CAM', fontsize=16)
    
    axes[0].imshow(image_pil)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam_image)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()