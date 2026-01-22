import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionHierarchicalCNN(nn.Module):
    def __init__(self, num_classes, numerical_feature_dim=47, dropout_rate=0.5):
        super().__init__()
        
        # --- Base CNN to create a larger feature map ---
        base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.features_extractor = nn.Sequential(
            base_cnn.conv1, base_cnn.bn1, base_cnn.relu, base_cnn.maxpool,
            base_cnn.layer1, base_cnn.layer2
        )
        base_feature_channels = 128

        # --- Processors for Global and Quadrant Levels ---
        self.global_processor = nn.Sequential(base_cnn.layer3, base_cnn.layer4, base_cnn.avgpool)
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1))
        )
        
        # --- Processor for Sub-Quadrants (Level 2) ---
        self.sub_quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1,1))
        )
        sub_quadrant_feature_dim = 64

        # --- NEW: Attention Gate for Sub-Quadrants ---
        self.attention_gate = nn.Sequential(
            nn.Linear(sub_quadrant_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs a single "importance score"
        )

        # --- Feature Dimensions ---
        total_image_feature_dim = 512 + (4 * 128) + sub_quadrant_feature_dim # Global + Quadrants + 1 Attended Sub-Quadrant vector

        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, 128),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate)
        )
        
        # --- Final Classifier ---
        combined_feature_dim = total_image_feature_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 1024),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )

    def forward(self, image_input, numerical_input):
        base_features = self.features_extractor(image_input)
        
        # --- Level 0 & 1 Processing (Global and Quadrants) ---
        global_f = self.global_processor(base_features).flatten(1)
        
        h, w = base_features.shape[2], base_features.shape[3]
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            base_features[:, :, :mid_h, :mid_w], base_features[:, :, :mid_h, mid_w:],
            base_features[:, :, mid_h:, :mid_w], base_features[:, :, mid_h:, mid_w:]
        ]
        quadrant_features = [self.quadrant_processor(q).flatten(1) for q in quadrants]

        # --- Level 2 Processing with Attention ---
        sub_quadrant_vectors = []
        for q in quadrants:
            qh, qw = q.shape[2], q.shape[3]
            mid_qh, mid_qw = qh // 2, qw // 2
            sub_quads = [
                q[:, :, :mid_qh, :mid_qw], q[:, :, :mid_qh, mid_qw:],
                q[:, :, mid_qh:, :mid_qw], q[:, :, mid_qh:, mid_qw:]
            ]
            sub_quadrant_vectors.extend([self.sub_quadrant_processor(sq).flatten(1) for sq in sub_quads])
        
        # Stack all 16 sub-quadrant vectors
        stacked_sub_quads = torch.stack(sub_quadrant_vectors, dim=1) # Shape: (B, 16, 64)

        # --- Apply Attention Gate ---
        # 1. Get an importance score for each of the 16 vectors
        attention_scores = self.attention_gate(stacked_sub_quads).squeeze(-1) # Shape: (B, 16)
        # 2. Convert scores to weights using Softmax
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1) # Shape: (B, 16, 1)
        # 3. Compute the weighted average of the sub-quadrant vectors
        attended_sub_quad_features = torch.sum(stacked_sub_quads * attention_weights, dim=1) # Shape: (B, 64)

        # --- Combine all image features ---
        all_image_features = [global_f] + quadrant_features + [attended_sub_quad_features]
        image_features = torch.cat(all_image_features, dim=1)

        # --- Process and Combine Numerical Features ---
        numerical_features = self.numerical_mlp(numerical_input)
        combined_features = torch.cat((image_features, numerical_features), dim=1)
        
        logits = self.classifier(combined_features)
        return logits


# --- (The other model classes like StandardMultimodalCNN, QuadtreeCNN, etc. remain the same) ---
class HierarchicalQuadtreeCNN(nn.Module):
    # ... (code for HierarchicalQuadtreeCNN remains the same)
    def __init__(self, num_classes, numerical_feature_dim=47, dropout_rate=0.5):
        super().__init__()
        
        # --- Base CNN to create a larger feature map ---
        base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 1. Initial Feature Extractor (stops at layer2 for a 28x28 map)
        self.features_extractor = nn.Sequential(
            base_cnn.conv1, base_cnn.bn1, base_cnn.relu, base_cnn.maxpool,
            base_cnn.layer1, base_cnn.layer2 # Output: (batch, 128, 28, 28)
        )
        base_feature_channels = 128

        # --- Define Processors for each of the 3 Levels ---

        # Level 0 Processor (Global)
        self.global_processor = nn.Sequential(
            base_cnn.layer3, # Process 28x28 -> 14x14
            base_cnn.layer4, # Process 14x14 -> 7x7
            base_cnn.avgpool # Process 7x7 -> 1x1
        )
        global_feature_dim = 512

        # Level 1 Processor (Quadrants: 14x14)
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)) # Pool to a single vector
        )
        quadrant_feature_dim = 128

        # Level 2 Processor (Sub-Quadrants: 7x7)
        self.sub_quadrant_processor = nn.Sequential(
            nn.Conv2d(base_feature_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)) # Pool to a single vector
        )
        sub_quadrant_feature_dim = 64

        # --- Fusion of Image Features ---
        # Total image features = Global + (4 * Quadrant) + (16 * Sub-Quadrant)
        total_image_feature_dim = global_feature_dim + (4 * quadrant_feature_dim) + (16 * sub_quadrant_feature_dim)

        # --- Numerical Feature Processor ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        numerical_output_dim = 128

        # --- Final Classifier ---
        combined_feature_dim = total_image_feature_dim + numerical_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )

    def forward(self, image_input, numerical_input):
        # Create the base 28x28 feature map
        base_features = self.features_extractor(image_input)

        # --- Level 0: Global Feature Extraction ---
        global_f = self.global_processor(base_features).flatten(1)

        # --- Level 1: Quadrant Feature Extraction ---
        h, w = base_features.shape[2], base_features.shape[3]
        mid_h, mid_w = h // 2, w // 2
        
        quadrants = [
            base_features[:, :, :mid_h, :mid_w],  # Top-Left
            base_features[:, :, :mid_h, mid_w:],  # Top-Right
            base_features[:, :, mid_h:, :mid_w],  # Bottom-Left
            base_features[:, :, mid_h:, w:]       # Bottom-Right
        ]
        
        quadrant_features = [self.quadrant_processor(q).flatten(1) for q in quadrants]

        # --- Level 2: Sub-Quadrant Feature Extraction ---
        sub_quadrant_features = []
        for q in quadrants:
            qh, qw = q.shape[2], q.shape[3]
            mid_qh, mid_qw = qh // 2, qw // 2
            sub_quads = [
                q[:, :, :mid_qh, :mid_qw],
                q[:, :, :mid_qh, mid_qw:],
                q[:, :, mid_qh:, :mid_qw],
                q[:, :, mid_qh:, qw:]
            ]
            sub_quadrant_features.extend([self.sub_quadrant_processor(sq).flatten(1) for sq in sub_quads])

        # --- Combine all image features ---
        all_features = [global_f] + quadrant_features + sub_quadrant_features
        image_features = torch.cat(all_features, dim=1)

        # --- Process and Combine Numerical Features ---
        numerical_features = self.numerical_mlp(numerical_input)
        combined_features = torch.cat((image_features, numerical_features), dim=1)

        # --- Final Classification ---
        logits = self.classifier(combined_features)
        return logits
class StandardMultimodalCNN(nn.Module):
    # ... (code for StandardMultimodalCNN remains the same)
    pass
class QuadtreeCNN(nn.Module):
    # ... (code for the original QuadtreeCNN remains the same)
    def __init__(self, num_classes, cnn_feature_dim=512, numerical_feature_dim=47, dropout_rate=0.5):
        super(QuadtreeCNN, self).__init__()

        # --- Branch 1: Image Processing with Quadtree CNN ---
        # Base CNN Backbone (ResNet18)
        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features_extractor = nn.Sequential(
            self.base_cnn.conv1,
            self.base_cnn.bn1,
            self.base_cnn.relu,
            self.base_cnn.maxpool,
            self.base_cnn.layer1, 
            self.base_cnn.layer2, 
            self.base_cnn.layer3  # Output: (batch, 256, 14, 14) for 224x224 input
        )
        
        quadrant_feature_channels = 256 
        
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(quadrant_feature_channels, cnn_feature_dim // 4, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # Output: (batch, 128, 3, 3) if input was 14x14
        )
        
        self.global_processor = nn.Sequential(
            self.base_cnn.layer4, # Output: (batch, 512, 7, 7)
            self.base_cnn.avgpool # Output: (batch, 512, 1, 1) -> flattened to (batch, 512)
        )
        
        # Calculate the output dimension of the image branch
        # Global features: 512
        # Each quadrant: 128 channels * 3x3 spatial = 1152 features
        # Total from 4 quadrants: 4 * 1152 = 4608
        # Total image features = 512 (global) + 4608 (quadrants) = 5120
        self.image_feature_dim = 512 + ( (cnn_feature_dim // 4) * 3 * 3 * 4 ) 
        assert self.image_feature_dim == 5120, f"Image feature dim mismatch: Expected 5120, got {self.image_feature_dim}"


        # --- Branch 2: Numerical Feature Processing (MLP) ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2), # Example expansion
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(numerical_feature_dim * 2, cnn_feature_dim // 2) # Project to a reasonable size, e.g., 256
        )
        self.numerical_output_dim = cnn_feature_dim // 2 # 256

        # --- Fusion and Classification Head ---
        self.combined_feature_dim = self.image_feature_dim + self.numerical_output_dim # 5120 + 256 = 5376

        self.classifier = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.combined_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.combined_feature_dim // 2, num_classes)
        )

    def forward(self, image_input, numerical_input):
        # --- Forward pass for Image Branch ---
        base_features = self.features_extractor(image_input) 
        
        h, w = base_features.shape[2], base_features.shape[3]
        
        q1 = base_features[:, :, :h//2, :w//2]   
        q2 = base_features[:, :, :h//2, w//2:]   
        q3 = base_features[:, :, h//2:, :w//2]   
        q4 = base_features[:, :, h//2:, w//2:]   

        q1_features = self.quadrant_processor(q1).flatten(1) 
        q2_features = self.quadrant_processor(q2).flatten(1)
        q3_features = self.quadrant_processor(q3).flatten(1)
        q4_features = self.quadrant_processor(q4).flatten(1)
        
        global_features = self.global_processor(base_features).flatten(1) 
        
        image_features = torch.cat(
            (global_features, q1_features, q2_features, q3_features, q4_features),
            dim=1 
        )

        # --- Forward pass for Numerical Branch ---
        numerical_features = self.numerical_mlp(numerical_input)

        # --- Concatenate (Fuse) Features from both branches ---
        combined_features = torch.cat((image_features, numerical_features), dim=1)
        
        # --- Final Classification ---
        logits = self.classifier(combined_features)
        
        return logits


# --- Updated Helper Function ---
def get_model(model_name, num_classes, device, print_num_params=True):
    model_name = model_name.lower()
    
    if model_name == 'quadtree':
        model = QuadtreeCNN(num_classes=num_classes).to(device)
    elif model_name == 'hierarchical_quadtree':
        model = HierarchicalQuadtreeCNN(num_classes=num_classes).to(device)
    elif model_name == 'attention_hierarchical': # New option
        model = AttentionHierarchicalCNN(num_classes=num_classes).to(device)
    else:
        model = StandardMultimodalCNN(backbone_name=model_name, num_classes=num_classes).to(device)

    if print_num_params:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: '{model_name.upper()}' | Trainable Parameters: {num_params / 1e6:.2f} Million")
    
    return model