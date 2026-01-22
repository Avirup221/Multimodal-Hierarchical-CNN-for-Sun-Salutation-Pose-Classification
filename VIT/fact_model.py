import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer
import json

class FactModel(nn.Module):
    """
    Fused Action-Conditioned Transformer (FACT) Model
    
    This model fuses visual tokens from a ViT backbone with
    numerical feature tokens (MediaPipe data) and passes them
    through a final Transformer for classification.
    """
    def __init__(self, num_classes, seq_len, num_numerical_features=47):
        super(FactModel, self).__init__()
        
        self.seq_len = seq_len
        self.num_numerical_features = num_numerical_features
        
        # --- 1. Vision Backbone (ViT) ---
        # Load a pre-trained Vision Transformer from timm
        # We will freeze its weights and use it as a feature extractor
        self.vit_backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k', 
            pretrained=True
        )
        self.vit_backbone.eval() # Set to evaluation mode
        
        # Freeze all parameters in the backbone
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
            
        self.embed_dim = self.vit_backbone.embed_dim # Should be 768
        
        # We are only interested in the patch tokens, not the [CLS] token
        # We will grab the output of the final 'blocks' layer
        self.vit_backbone.head = nn.Identity() # Remove the final classification head
        
        # --- 2. Numerical Feature Processor ---
        # An MLP to project numerical features to the same dimension as image tokens
        self.numerical_projector = nn.Sequential(
            nn.Linear(self.num_numerical_features, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, self.embed_dim)
        )
        
        # --- 3. Final Fusion Transformer ---
        # We'll use a few standard TransformerEncoder layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=8, 
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True # Expects (Batch, Seq, Features)
        )
        self.fusion_transformer = nn.TransformerEncoder(transformer_layer, num_layers=4)
        
        # --- 4. Special Tokens ---
        # A global [CLS] token for the final classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        
        # Positional Embeddings for the *entire* fused sequence
        # Seq = (SEQ_LEN * num_patches) + (SEQ_LEN * 1 numerical_token) + 1 [CLS] token
        # Let's simplify: We'll average-pool the ViT patches for each frame
        # New Seq = SEQ_LEN (image tokens) + SEQ_LEN (numerical tokens) + 1 [CLS] token
        #
        # --- SIMPLIFIED (BETTER) APPROACH ---
        # We will use the ViT to get ONE feature vector per image (the [CLS] token output)
        # This is much more memory efficient.
        
        # 1. Re-define ViT backbone (we *do* want the [CLS] token output)
        self.vit_backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k', 
            pretrained=True,
            num_classes=0 # This gives us the [CLS] token embedding directly
        )
        self.vit_backbone.eval()
        for param in self.vit_backbone.parameters():
            param.requires_grad = False
            
        # 2. Positional Embeddings
        # We will have (SEQ_LEN * 2) tokens + 1 [CLS] token
        # (1 image token + 1 num token) for each of the SEQ_LEN frames
        self.pos_embed = nn.Parameter(torch.randn(1, (self.seq_len * 2) + 1, self.embed_dim))
        
        # 3. Token Type Embeddings (to distinguish image vs. numerical)
        self.token_type_embed = nn.Embedding(2, self.embed_dim) # 0 = image, 1 = numerical
        
        # --- 5. Classification Head ---
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, image_sequence, numerical_sequence):
        # image_sequence: (B, T, C, H, W)
        # numerical_sequence: (B, T, F_num)
        batch_size = image_sequence.shape[0]
        
        # 1. Process Image Sequence
        # We need to run the ViT on each frame individually
        # Reshape (B, T, C, H, W) -> (B*T, C, H, W)
        image_input = image_sequence.view(batch_size * self.seq_len, 3, 224, 224)
        
        # Get (B*T, 768) feature vectors from the [CLS] token
        # Use with torch.no_grad() for safety, though params are frozen
        with torch.no_grad():
            image_tokens = self.vit_backbone(image_input) # (B*T, 768)
        
        # Reshape back to (B, T, 768)
        image_tokens = image_tokens.view(batch_size, self.seq_len, self.embed_dim)
        
        # 2. Process Numerical Sequence
        # (B, T, F_num) -> (B, T, 768)
        numerical_tokens = self.numerical_projector(numerical_sequence)
        
        # 3. Create Fused Token Sequence
        # Add token type embeddings
        image_tokens = image_tokens + self.token_type_embed(torch.zeros(1, 1, dtype=torch.long, device=image_tokens.device))
        numerical_tokens = numerical_tokens + self.token_type_embed(torch.ones(1, 1, dtype=torch.long, device=numerical_tokens.device))
        
        # Interleave the tokens: [img1, num1, img2, num2, ...]
        fused_sequence = torch.stack((image_tokens, numerical_tokens), dim=2) # (B, T, 2, 768)
        fused_sequence = fused_sequence.view(batch_size, self.seq_len * 2, self.embed_dim) # (B, T*2, 768)
        
        # 4. Add [CLS] token and Positional Embeddings
        # Expand CLS token to match batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, 768)
        
        # Prepend CLS token to the sequence
        full_sequence = torch.cat((cls_tokens, fused_sequence), dim=1) # (B, T*2 + 1, 768)
        
        # Add positional embeddings
        full_sequence = full_sequence + self.pos_embed
        
        # 5. Run through Fusion Transformer
        transformer_output = self.fusion_transformer(full_sequence) # (B, T*2 + 1, 768)
        
        # 6. Classify
        # Get the output corresponding to the [CLS] token (the first token)
        cls_output = transformer_output[:, 0] # (B, 768)
        
        # Pass through the final classification head
        logits = self.classification_head(cls_output) # (B, num_classes)
        
        return logits