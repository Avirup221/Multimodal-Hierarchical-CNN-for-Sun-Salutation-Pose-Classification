import torch
import torch.nn as nn
import torchvision.models as models

# --- Helper: Basic 3D Convolutional Block ---
def conv_3d_block(in_channels, out_channels, kernel_size=(3, 3, 3), stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )

# --- Model 1: CNN + LSTM (Multimodal) ---
class CnnLstm(nn.Module):
    def __init__(self, num_classes, sequence_length=4, numerical_feature_dim=47, dropout_rate=0.5, lstm_hidden_size=256):
        super(CnnLstm, self).__init__()
        self.sequence_length = sequence_length

        # --- Branch 1: Image (CNN) ---
        # Use ResNet-18 backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove the final FC layer. Output dim is 512.
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze CNN weights (optional, but good for speed/stability)
        for param in self.cnn_backbone.parameters():
            param.requires_grad = False
            
        cnn_output_dim = 512

        # --- Branch 2: Numerical (MLP) ---
        self.numerical_mlp = nn.Sequential(
            nn.Linear(numerical_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128) # Project to 128 dim
        )
        numerical_out_dim = 128

        # --- Temporal Fusion (LSTM) ---
        # We fuse features at each timestep *before* the LSTM
        lstm_input_dim = cnn_output_dim + numerical_out_dim
        
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_sequence, numerical_sequence):
        # image_sequence: (B, T, C, H, W)
        # numerical_sequence: (B, T, F_num)
        batch_size, seq_len, c, h, w = image_sequence.shape
        
        # 1. Process Images (CNN)
        # Reshape to (B*T, C, H, W) to process all frames at once
        c_in = image_sequence.view(batch_size * seq_len, c, h, w)
        
        # Run CNN: Output is (B*T, 512, 1, 1)
        c_out = self.cnn_backbone(c_in)
        c_out = c_out.view(batch_size, seq_len, -1) # (B, T, 512)
        
        # 2. Process Numerical Data
        # (B, T, F_num) -> (B, T, 128)
        n_out = self.numerical_mlp(numerical_sequence)
        
        # 3. Fuse Features
        # Concatenate along the feature dimension
        fused_input = torch.cat((c_out, n_out), dim=2) # (B, T, 512+128)
        
        # 4. Run LSTM
        # Output is (B, T, hidden_size)
        lstm_out, _ = self.lstm(fused_input)
        
        # We take the output of the LAST time step for classification
        final_state = lstm_out[:, -1, :] # (B, hidden_size)
        
        # 5. Classify
        logits = self.classifier(final_state)
        return logits


# --- Model 2: 3D CNN (Ji et al. Style) ---
class Ji3DCNN(nn.Module):
    def __init__(self, num_classes, sequence_length=4, numerical_feature_dim=47, dropout_rate=0.5):
        super(Ji3DCNN, self).__init__()
        
        # --- Visual Stream (3D Conv) ---
        # Input: (B, 3, T, H, W)
        self.visual_stream = nn.Sequential(
            conv_3d_block(3, 32, kernel_size=(3,3,3), padding=1),
            nn.MaxPool3d(kernel_size=(1, 2, 2)), # Pool spatial only
            conv_3d_block(32, 64, kernel_size=(3,3,3), padding=1),
            nn.MaxPool3d(kernel_size=(2, 2, 2)), # Pool T and spatial
            conv_3d_block(64, 128, kernel_size=(3,3,3), padding=1),
            nn.AdaptiveAvgPool3d((1, 1, 1)) # Global Pool -> (B, 128, 1, 1, 1)
        )
        visual_out_dim = 128
        
        # --- Numerical Stream (LSTM) ---
        # We use a small LSTM for the numerical data
        self.numerical_lstm = nn.LSTM(
            input_size=numerical_feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        numerical_out_dim = 64
        
        # --- Fusion & Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(visual_out_dim + numerical_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_sequence, numerical_sequence):
        # image_sequence: (B, T, 3, H, W) -> Needs (B, 3, T, H, W)
        # numerical_sequence: (B, T, F_num)
        
        # 1. Visual Stream
        v_in = image_sequence.permute(0, 2, 1, 3, 4) # Swap T and C
        v_out = self.visual_stream(v_in).flatten(1) # (B, 128)
        
        # 2. Numerical Stream
        lstm_out, _ = self.numerical_lstm(numerical_sequence)
        n_out = lstm_out[:, -1, :] # Last step: (B, 64)
        
        # 3. Fuse & Classify
        fused = torch.cat((v_out, n_out), dim=1)
        logits = self.classifier(fused)
        return logits

# --- Helper Function ---
def get_model(model_name, num_classes, device, seq_len=4, num_features=47):
    if model_name == 'cnn_lstm':
        model = CnnLstm(num_classes, sequence_length=seq_len, numerical_feature_dim=num_features)
    elif model_name == '3d_cnn':
        model = Ji3DCNN(num_classes, sequence_length=seq_len, numerical_feature_dim=num_features)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model.to(device)