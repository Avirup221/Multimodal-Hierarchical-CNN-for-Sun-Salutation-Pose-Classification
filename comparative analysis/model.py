import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

# ------------------- NEW GENERIC MULTIMODAL MODEL -------------------
class StandardMultimodalCNN(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True, numerical_feature_dim=47, dropout_rate=0.5):
        super().__init__()

        # --- Branch 1: Image Processing with a Standard CNN Backbone ---
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_output_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity() # Remove the final classification layer
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_output_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'vgg16':
            weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.vgg16(weights=weights)
            backbone_output_features = self.backbone.classifier[0].in_features # VGG's features are in classifier
            self.backbone.classifier = nn.Identity() # Remove the entire classifier block
        elif backbone_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
            backbone_output_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'densenet121':
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.densenet121(weights=weights)
            backbone_output_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone '{backbone_name}' not supported.")

        # --- Branch 2: Numerical Feature Processing (MLP) ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(numerical_feature_dim * 2, 256) # Project to a fixed size
        )
        numerical_output_dim = 256

        # --- Fusion and Classification Head ---
        combined_feature_dim = backbone_output_features + numerical_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, image_input, numerical_input):
        # Forward pass for Image Branch
        image_features = self.backbone(image_input)

        # Forward pass for Numerical Branch
        numerical_features = self.numerical_mlp(numerical_input)

        # Concatenate (Fuse) Features
        combined_features = torch.cat((image_features, numerical_features), dim=1)
        
        # Final Classification
        logits = self.classifier(combined_features)
        return logits

# ------------------- ORIGINAL QUADTREECNN (for reference) -------------------
class QuadtreeCNN(nn.Module):
    def __init__(self, num_classes, cnn_feature_dim=512, numerical_feature_dim=47, dropout_rate=0.5):
        super(QuadtreeCNN, self).__init__()
        # --- Branch 1: Image Processing with Quadtree CNN ---
        self.base_cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.features_extractor = nn.Sequential(
            self.base_cnn.conv1, self.base_cnn.bn1, self.base_cnn.relu, self.base_cnn.maxpool,
            self.base_cnn.layer1, self.base_cnn.layer2, self.base_cnn.layer3
        )
        quadrant_feature_channels = 256
        self.quadrant_processor = nn.Sequential(
            nn.Conv2d(quadrant_feature_channels, cnn_feature_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.global_processor = nn.Sequential(self.base_cnn.layer4, self.base_cnn.avgpool)
        self.image_feature_dim = 512 + ((cnn_feature_dim // 4) * 3 * 3 * 4)
        
        # --- Branch 2: Numerical Feature Processing (MLP) ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, numerical_feature_dim * 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(numerical_feature_dim * 2, cnn_feature_dim // 2)
        )
        self.numerical_output_dim = cnn_feature_dim // 2
        
        # --- Fusion and Classification Head ---
        self.combined_feature_dim = self.image_feature_dim + self.numerical_output_dim
        self.classifier = nn.Sequential(
            nn.Linear(self.combined_feature_dim, self.combined_feature_dim // 2),
            nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(self.combined_feature_dim // 2, num_classes)
        )

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

# ------------------- UPDATED HELPER FUNCTION -------------------
def get_model(model_name, num_classes, device, print_num_params=True):
    """
    Initializes a model based on the provided name.
    Supported names: 'quadtree', 'resnet18', 'resnet50', 'vgg16', 'mobilenet_v2', 'densenet121'
    """
    model_name = model_name.lower()
    
    if model_name == 'quadtree':
        model = QuadtreeCNN(num_classes=num_classes, numerical_feature_dim=47).to(device)
    else:
        model = StandardMultimodalCNN(backbone_name=model_name, num_classes=num_classes).to(device)

    if print_num_params:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: '{model_name.upper()}' | Trainable Parameters: {num_params / 1e6:.2f} Million")
    
    return model