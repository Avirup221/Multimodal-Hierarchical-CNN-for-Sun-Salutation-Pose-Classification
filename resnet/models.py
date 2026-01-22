import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models 

# --- NEW MODEL CLASS: StandardResNetCNN ---
class StandardResNetCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(StandardResNetCNN, self).__init__()
        
        # Use pre-trained ResNet18 as the backbone
        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze all layers of ResNet18 initially for feature extraction
        for param in self.base_cnn.parameters():
            param.requires_grad = False

        # Define the feature extractor path up to the last convolutional layer (layer4)
        # This is the output we'll target for Grad-CAM
        self.features_extractor = nn.Sequential(
            self.base_cnn.conv1,
            self.base_cnn.bn1,
            self.base_cnn.relu,
            self.base_cnn.maxpool,
            self.base_cnn.layer1,
            self.base_cnn.layer2,
            self.base_cnn.layer3,
            self.base_cnn.layer4 # Output of layer4 is (batch, 512, 7, 7) for 224x224 input
        )
        
        # Use ResNet's original adaptive average pooling
        self.avgpool = self.base_cnn.avgpool 

        # Final classification head
        # Input to classifier will be 512 (from avgpool)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), # Hidden layer
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes) # Output layer
        )

        # --- Grad-CAM Specific Attributes (same as QuadtreeCNN for consistency) ---
        self.gradients = None
        self.activations = None

    # Hook function to capture gradients
    def save_gradient_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple, we want the gradient w.r.t the output of the module
        self.gradients = grad_output[0] 

    # Hook function to capture activations
    def save_activation_hook(self, module, input, output):
        self.activations = output

    def forward(self, image_input, numerical_input=None): # numerical_input is ignored for this model
        # Get activations up to layer4. This is the tensor we will register hooks on
        features_pre_avgpool = self.features_extractor(image_input)
        
        # Apply avgpool and flatten
        features = self.avgpool(features_pre_avgpool).flatten(1)
        
        # Final classification
        logits = self.classifier(features)
        return logits

# --- END NEW MODEL CLASS: StandardResNetCNN ---


class QuadtreeCNN(nn.Module):
    def __init__(self, num_classes, cnn_feature_dim=512, numerical_feature_dim=47, dropout_rate=0.5, mode='fusion'):
        super(QuadtreeCNN, self).__init__()
        
        self.mode = mode 

        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.base_cnn.parameters():
            param.requires_grad = False

        self.features_extractor = nn.Sequential(
            self.base_cnn.conv1,
            self.base_cnn.bn1,
            self.base_cnn.relu,
            self.base_cnn.maxpool,
            self.base_cnn.layer1, 
            self.base_cnn.layer2, 
            self.base_cnn.layer3  
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


        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(numerical_feature_dim * 2, cnn_feature_dim // 2) 
        )
        self.numerical_output_dim = cnn_feature_dim // 2 

        if self.mode == 'fusion':
            self.final_classifier_input_dim = self.image_feature_dim + self.numerical_output_dim 
        elif self.mode == 'image_only':
            self.final_classifier_input_dim = self.image_feature_dim 
        elif self.mode == 'numerical_only':
            self.final_classifier_input_dim = self.numerical_output_dim 
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'fusion', 'image_only', 'numerical_only', 'standard_resnet_only'.")

        self.classifier = nn.Sequential(
            nn.Linear(self.final_classifier_input_dim, self.final_classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_classifier_input_dim // 2, num_classes)
        )

        # --- Grad-CAM Specific Attributes ---
        self.gradients = None
        self.activations = None

    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] 

    def save_activation_hook(self, module, input, output):
        self.activations = output

    def forward(self, image_input, numerical_input):
        image_features = None
        numerical_features = None

        if self.mode in ['fusion', 'image_only']:
            base_features = self.features_extractor(image_input) 
            
            global_features_pre_avgpool = self.base_cnn.layer4(base_features)
            global_features = self.base_cnn.avgpool(global_features_pre_avgpool).flatten(1)

            h, w = base_features.shape[2], base_features.shape[3]
            
            q1 = base_features[:, :, :h//2, :w//2]   
            q2 = base_features[:, :, :h//2, w//2:]   
            q3 = base_features[:, :, h//2:, :w//2]   
            q4 = base_features[:, :, h//2:, w//2:]   

            q1_features = self.quadrant_processor(q1).flatten(1) 
            q2_features = self.quadrant_processor(q2).flatten(1)
            q3_features = self.quadrant_processor(q3).flatten(1)
            q4_features = self.quadrant_processor(q4).flatten(1)
            
            image_features = torch.cat(
                (global_features, q1_features, q2_features, q3_features, q4_features),
                dim=1 
            )

        if self.mode in ['fusion', 'numerical_only']:
            numerical_features = self.numerical_mlp(numerical_input)

        if self.mode == 'fusion':
            combined_features = torch.cat((image_features, numerical_features), dim=1)
        elif self.mode == 'image_only':
            combined_features = image_features
        elif self.mode == 'numerical_only':
            combined_features = numerical_features
        
        logits = self.classifier(combined_features)
        
        return logits


def get_model(num_classes, device, numerical_feature_dim=47, mode='fusion', print_num_params=True):
    # --- MODIFIED: Instantiate different models based on mode ---
    if mode == 'standard_resnet_only':
        model = StandardResNetCNN(num_classes=num_classes).to(device)
    else:
        model = QuadtreeCNN(num_classes=num_classes, numerical_feature_dim=numerical_feature_dim, mode=mode).to(device)

    if print_num_params:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f} Million (Mode: {mode})")
    
    return model

