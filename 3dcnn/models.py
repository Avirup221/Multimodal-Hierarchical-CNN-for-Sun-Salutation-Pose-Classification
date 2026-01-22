import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models 
import torchvision.models.video as video_models 

# --- NEW LOSS CLASS: FocalLoss (remains, not used by default in train script) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', num_classes=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes 

        if isinstance(self.alpha, (float, int)):
            self.alpha = torch.tensor([self.alpha, 1 - self.alpha]) 
        elif isinstance(self.alpha, list):
            self.alpha = torch.tensor(self.alpha)

    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=-1) 
        pt = torch.exp(log_pt) 

        pt = pt.gather(1, targets.view(-1, 1)).squeeze() 
        log_pt = log_pt.gather(1, targets.view(-1, 1)).squeeze() 

        loss_modulating_factor = (1 - pt).pow(self.gamma) 

        if self.alpha.dim() > 1: 
            alpha_t = self.alpha[targets] 
        elif self.alpha.dim() == 1 and self.alpha.size(0) == self.num_classes:
            alpha_t = self.alpha.to(targets.device)[targets] 
        else: 
            if self.alpha.dim() == 0: 
                alpha_t = alpha_t.expand_as(loss_modulating_factor)

        loss = -alpha_t * loss_modulating_factor * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else: 
            return loss


# --- EXISTING MODEL CLASS: StandardResNetCNN ---
class StandardResNetCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(StandardResNetCNN, self).__init__()
        
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
            self.base_cnn.layer3,
            self.base_cnn.layer4 
        )
        
        self.avgpool = self.base_cnn.avgpool 

        self.classifier = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes) 
        )

        self.gradients = None
        self.activations = None

    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] 

    def save_activation_hook(self, module, input, output):
        self.activations = output

    def forward(self, image_input, numerical_input=None): 
        features_pre_avgpool = self.features_extractor(image_input)
        features = self.avgpool(features_pre_avgpool).flatten(1)
        logits = self.classifier(features)
        return logits

# --- END StandardResNetCNN ---


# --- UPDATED MODEL CLASS: Quadtree3DCNN (Increased Dropout) ---
class Quadtree3DCNN(nn.Module):
    def __init__(self, num_classes, sequence_length=8, cnn_3d_feature_dim=1024, numerical_feature_dim=47, dropout_rate=0.6, mode='quadtree_3d_fusion'):
        super(Quadtree3DCNN, self).__init__()
        
        self.mode = mode
        self.sequence_length = sequence_length
        self.cnn_3d_feature_dim = cnn_3d_feature_dim 
        self.numerical_feature_dim = numerical_feature_dim

        # --- 3D CNN Backbone (Image Sequence Processing) ---
        # This is your custom 3D CNN backbone
        self.conv3d_block1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        )

        self.conv3d_block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) 
        )

        self.conv3d_block3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)) 
        )
        
        self.conv3d_block4_new = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) 
        )

        self.conv3d_final_features = nn.Sequential(
            nn.Conv3d(256, self.cnn_3d_feature_dim, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.cnn_3d_feature_dim),
            nn.ReLU(inplace=True)
        )

        self.global_avg_pool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        # --- Numerical Sequence Processing (LSTM) ---
        self.numerical_lstm = nn.LSTM(
            input_size=numerical_feature_dim,
            hidden_size=numerical_feature_dim * 4, 
            num_layers=2,
            batch_first=True, 
            dropout=dropout_rate if 2 > 1 else 0 
        )
        self.numerical_lstm_output_dim = numerical_feature_dim * 4 
        
        self.numerical_projection = nn.Sequential(
            nn.Linear(self.numerical_lstm_output_dim, self.cnn_3d_feature_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.numerical_final_dim = self.cnn_3d_feature_dim // 2

        # --- Final Classifier ---
        if self.mode == 'quadtree_3d_fusion':
            self.final_classifier_input_dim = self.cnn_3d_feature_dim + self.numerical_final_dim
        elif self.mode == 'quadtree_3d_image_only':
            self.final_classifier_input_dim = self.cnn_3d_feature_dim
        else:
            raise ValueError(f"Invalid mode for Quadtree3DCNN: {self.mode}. Choose from 'quadtree_3d_fusion', 'quadtree_3d_image_only'.")

        self.classifier = nn.Sequential(
            nn.Linear(self.final_classifier_input_dim, self.final_classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_classifier_input_dim // 2, num_classes)
        )

        self.gradients = None
        self.activations = None

    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] 

    def save_activation_hook(self, module, input, output):
        self.activations = output

    def forward(self, image_sequence_input, numerical_sequence_input):
        image_features = None
        numerical_features = None

        if self.mode in ['quadtree_3d_fusion', 'quadtree_3d_image_only']:
            image_sequence_input_permuted = image_sequence_input.permute(0, 2, 1, 3, 4) 

            x = self.conv3d_block1(image_sequence_input_permuted)
            x = self.conv3d_block2(x)
            x = self.conv3d_block3(x)
            x = self.conv3d_block4_new(x)
            
            x = self.conv3d_final_features(x) 
            
            image_features = self.global_avg_pool_3d(x).flatten(1) 

        if self.mode in ['quadtree_3d_fusion']: 
            lstm_out, _ = self.numerical_lstm(numerical_sequence_input)
            numerical_features_last_timestep = lstm_out[:, -1, :] 
            numerical_features = self.numerical_projection(numerical_features_last_timestep) 

        if self.mode == 'quadtree_3d_fusion':
            combined_features = torch.cat((image_features, numerical_features), dim=1)
        elif self.mode == 'quadtree_3d_image_only':
            combined_features = image_features
        else:
            raise ValueError(f"Invalid mode during forward pass: {self.mode}")
        
        logits = self.classifier(combined_features)
        
        return logits

# --- END UPDATED MODEL CLASS: Quadtree3DCNN ---


# --- EXISTING MODEL CLASS: ResNet3DVideo (Modified for Fine-tuning) ---
class ResNet3DVideo(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ResNet3DVideo, self).__init__()
        # Load a pre-trained R3D_18 model
        self.r3d_model = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
        
        # --- MODIFIED: Selective Unfreezing for Fine-tuning ---
        # Freeze all parameters initially
        for param in self.r3d_model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers for fine-tuning.
        # It's common to unfreeze later layers first, as they learn more high-level features.
        # You can experiment with unfreezing more layers (e.g., layer3, layer2)
        # but start with layer4 and the final classification head.
        for param in self.r3d_model.layer4.parameters():
            param.requires_grad = True
        print("Unfrozen r3d_18.layer4 for fine-tuning.")
        
        # --- END MODIFIED ---

        # Replace the final classification head
        num_ftrs = self.r3d_model.fc.in_features
        self.r3d_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_ftrs // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs // 2, num_classes)
        )
        # Ensure the new classification head is trainable
        for param in self.r3d_model.fc.parameters():
            param.requires_grad = True
        print("Replaced and unfrozen r3d_18.fc (classifier).")


    def forward(self, image_sequence_input, numerical_input=None):
        image_sequence_input_permuted = image_sequence_input.permute(0, 2, 1, 3, 4)
        
        logits = self.r3d_model(image_sequence_input_permuted)
        return logits

# --- END EXISTING MODEL CLASS: ResNet3DVideo ---


# --- NEW MODEL CLASS: HybridQuadtree3DCNN (Integrating Pre-trained R3D_18 Backbone) ---
class HybridQuadtree3DCNN(nn.Module):
    def __init__(self, num_classes, sequence_length=8, numerical_feature_dim=47, dropout_rate=0.6, mode='hybrid_quadtree_3d_fusion'):
        super(HybridQuadtree3DCNN, self).__init__()
        
        self.mode = mode
        self.sequence_length = sequence_length
        self.numerical_feature_dim = numerical_feature_dim

        # --- Pre-trained 3D CNN Backbone (Image Sequence Processing) ---
        # Load a pre-trained R3D_18 model and use its feature extractor
        r3d_base = video_models.r3d_18(weights=video_models.R3D_18_Weights.KINETICS400_V1)
        
        # Extract the feature-extracting layers (excluding the final average pool and FC layer)
        # The sequential structure of r3d_18 is:
        # avgpool (AdaptiveAvgPool3d)
        # fc (Linear)
        # We want everything before avgpool.
        self.pretrained_image_extractor = nn.Sequential(
            r3d_base.stem,
            r3d_base.layer1,
            r3d_base.layer2,
            r3d_base.layer3,
            r3d_base.layer4
        )
        
        # Freeze all parameters of the pre-trained backbone initially
        for param in self.pretrained_image_extractor.parameters():
            param.requires_grad = False

        # Unfreeze layer4 for fine-tuning, as it's common practice for better adaptation
        for param in self.pretrained_image_extractor[4].parameters(): # layer4 is the 5th module (index 4) in Sequential
            param.requires_grad = True
        print("Unfrozen r3d_18.layer4 within HybridQuadtree3DCNN for fine-tuning.")

        # The output feature dimension of r3d_18's layer4 before avgpool is 512 channels.
        # After global average pooling, it will be 512 features.
        self.cnn_3d_feature_dim = 512 
        self.global_avg_pool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))


        # --- Numerical Sequence Processing (LSTM) - Same as original Quadtree3DCNN ---
        self.numerical_lstm = nn.LSTM(
            input_size=numerical_feature_dim,
            hidden_size=numerical_feature_dim * 4, 
            num_layers=2,
            batch_first=True, 
            dropout=dropout_rate if 2 > 1 else 0 
        )
        self.numerical_lstm_output_dim = numerical_feature_dim * 4 
        
        self.numerical_projection = nn.Sequential(
            nn.Linear(self.numerical_lstm_output_dim, self.cnn_3d_feature_dim // 2), 
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.numerical_final_dim = self.cnn_3d_feature_dim // 2

        # --- Final Classifier ---
        if self.mode == 'hybrid_quadtree_3d_fusion':
            self.final_classifier_input_dim = self.cnn_3d_feature_dim + self.numerical_final_dim
        elif self.mode == 'hybrid_quadtree_3d_image_only': # Option for image-only with this hybrid
            self.final_classifier_input_dim = self.cnn_3d_feature_dim
        else:
            raise ValueError(f"Invalid mode for HybridQuadtree3DCNN: {self.mode}. Choose from 'hybrid_quadtree_3d_fusion', 'hybrid_quadtree_3d_image_only'.")

        self.classifier = nn.Sequential(
            nn.Linear(self.final_classifier_input_dim, self.final_classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_classifier_input_dim // 2, num_classes)
        )

        self.gradients = None
        self.activations = None

    def save_gradient_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] 

    def save_activation_hook(self, module, input, output):
        self.activations = output

    def forward(self, image_sequence_input, numerical_sequence_input):
        image_features = None
        numerical_features = None

        if self.mode in ['hybrid_quadtree_3d_fusion', 'hybrid_quadtree_3d_image_only']:
            # r3d_18 expects input in (Batch, Channels, Depth/Time, Height, Width) format
            # Your dataloader provides (Batch, Time, Channels, Height, Width)
            # So, permute (B, T, C, H, W) to (B, C, T, H, W)
            image_sequence_input_permuted = image_sequence_input.permute(0, 2, 1, 3, 4) 
            
            # Pass through the pre-trained R3D backbone
            x = self.pretrained_image_extractor(image_sequence_input_permuted)
            
            image_features = self.global_avg_pool_3d(x).flatten(1) 

        if self.mode in ['hybrid_quadtree_3d_fusion']: 
            lstm_out, _ = self.numerical_lstm(numerical_sequence_input)
            numerical_features_last_timestep = lstm_out[:, -1, :] 
            numerical_features = self.numerical_projection(numerical_features_last_timestep) 

        if self.mode == 'hybrid_quadtree_3d_fusion':
            combined_features = torch.cat((image_features, numerical_features), dim=1)
        elif self.mode == 'hybrid_quadtree_3d_image_only':
            combined_features = image_features
        else:
            raise ValueError(f"Invalid mode during forward pass for HybridQuadtree3DCNN: {self.mode}")
        
        logits = self.classifier(combined_features)
        
        return logits

# --- END NEW MODEL CLASS: HybridQuadtree3DCNN ---


# --- EXISTING MODEL CLASS: QuadtreeCNN ---
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
            raise ValueError(f"Invalid mode: {self.mode}. Choose from 'fusion', 'image_only', 'numerical_only', 'standard_resnet_only', 'quadtree_3d_fusion', 'quadtree_3d_image_only'.")

        self.classifier = nn.Sequential(
            nn.Linear(self.final_classifier_input_dim, self.final_classifier_input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_classifier_input_dim // 2, num_classes)
        )

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


def get_model(num_classes, device, numerical_feature_dim=47, mode='fusion', sequence_length=8, print_num_params=True):
    if mode == 'standard_resnet_only':
        model = StandardResNetCNN(num_classes=num_classes).to(device)
    elif mode in ['quadtree_3d_fusion', 'quadtree_3d_image_only']:
        model = Quadtree3DCNN(
            num_classes=num_classes, 
            sequence_length=sequence_length, 
            numerical_feature_dim=numerical_feature_dim, 
            mode=mode,
            cnn_3d_feature_dim=1024 
        ).to(device)
    elif mode == 'resnet_3d_video_only':
        model = ResNet3DVideo(num_classes=num_classes).to(device)
    # --- NEW: Instantiate HybridQuadtree3DCNN model ---
    elif mode in ['hybrid_quadtree_3d_fusion', 'hybrid_quadtree_3d_image_only']:
        model = HybridQuadtree3DCNN(
            num_classes=num_classes, 
            sequence_length=sequence_length, 
            numerical_feature_dim=numerical_feature_dim, 
            mode=mode
        ).to(device)
    # --- END NEW ---
    else:
        model = QuadtreeCNN(num_classes=num_classes, numerical_feature_dim=numerical_feature_dim, mode=mode).to(device)

    if print_num_params:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {num_params / 1e6:.2f} Million (Mode: {mode})")
    
    return model

