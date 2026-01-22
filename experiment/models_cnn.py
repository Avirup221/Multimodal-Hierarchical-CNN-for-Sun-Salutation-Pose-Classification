import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models # For CNN backbone

class QuadtreeCNN(nn.Module):
    def __init__(self, num_classes, cnn_feature_dim=512, numerical_feature_dim=47, dropout_rate=0.5, mode='fusion'):
        super(QuadtreeCNN, self).__init__()
        
        self.mode = mode # 'fusion', 'image_only', 'numerical_only'

        # --- Branch 1: Image Processing with Quadtree CNN ---
        # Base CNN Backbone (ResNet18)
        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Freeze all layers of ResNet18 initially for feature extraction
        for param in self.base_cnn.parameters():
            param.requires_grad = False

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
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        self.global_processor = nn.Sequential(
            self.base_cnn.layer4, 
            self.base_cnn.avgpool 
        )
        
        self.image_feature_dim = 512 + ( (cnn_feature_dim // 4) * 3 * 3 * 4 ) 
        assert self.image_feature_dim == 5120, f"Image feature dim mismatch: Expected 5120, got {self.image_feature_dim}"


        # --- Branch 2: Numerical Feature Processing (MLP) ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(numerical_feature_dim * 2, cnn_feature_dim // 2) 
        )
        self.numerical_output_dim = cnn_feature_dim // 2 # 256

        # --- Determine final classifier input dimension based on mode ---
        if self.mode == 'fusion':
            self.final_classifier_input_dim = self.image_feature_dim + self.numerical_output_dim # 5120 + 256 = 5376
        elif self.mode == 'image_only':
            self.final_classifier_input_dim = self.image_feature_dim # 5120
        elif self.mode == 'numerical_only':
            self.final_classifier_input_dim = self.numerical_output_dim # 256
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'fusion', 'image_only', 'numerical_only'.")

        # --- Final Classification Head ---
        self.classifier = nn.Sequential(
            nn.Linear(self.final_classifier_input_dim, self.final_classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_classifier_input_dim // 2, num_classes)
        )

    def forward(self, image_input, numerical_input):
        # Initialize features to None, then populate based on mode
        image_features = None
        numerical_features = None

        if self.mode in ['fusion', 'image_only']:
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

        if self.mode in ['fusion', 'numerical_only']:
            # --- Forward pass for Numerical Branch ---
            numerical_features = self.numerical_mlp(numerical_input)

        # --- Concatenate (Fuse) Features based on mode ---
        if self.mode == 'fusion':
            combined_features = torch.cat((image_features, numerical_features), dim=1)
        elif self.mode == 'image_only':
            combined_features = image_features
        elif self.mode == 'numerical_only':
            combined_features = numerical_features
        
        # --- Final Classification ---
        logits = self.classifier(combined_features)
        
        return logits


# Helper function to get the model
def get_model(num_classes, device, numerical_feature_dim=47, mode='fusion', print_num_params=True):
    model = QuadtreeCNN(num_classes=num_classes, numerical_feature_dim=numerical_feature_dim, mode=mode).to(device)

    if print_num_params:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f} Million (Mode: {mode})")
    
    return model

