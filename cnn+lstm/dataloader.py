import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# --- 1. CONFIGURATION ---
DATA_DIR = "/home/avirupd/summer_project/new_sequential_dataset_seq4_v3"
SEQ_LEN = 4
# --- END CONFIGURATION ---

class YogaPoseSequentialDataset(Dataset):
    """
    Loads .pt files from the sequential dataset.
    """
    def __init__(self, data_dir, split, class_to_idx):
        self.data_dir = os.path.join(data_dir, split)
        self.class_to_idx = class_to_idx
        self.file_paths = []
        
        if not os.path.exists(self.data_dir):
            print(f"Warning: Directory not found, skipping: {self.data_dir}")
            return
            
        for class_name, idx in self.class_to_idx.items():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            
            for f in os.listdir(class_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(class_dir, f))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = torch.load(file_path)
            
            img_seq = data['image_sequence']
            num_seq = data['numerical_sequence']
            label = data['label']
            
            # --- Data Consistency Check ---
            if img_seq.shape[0] != SEQ_LEN:
                if img_seq.shape[0] > SEQ_LEN:
                    img_seq = img_seq[:SEQ_LEN]
                else:
                    padding = img_seq[-1].unsqueeze(0).repeat(SEQ_LEN - img_seq.shape[0], 1, 1, 1)
                    img_seq = torch.cat((img_seq, padding), dim=0)
            
            if num_seq.shape[0] != SEQ_LEN:
                if num_seq.shape[0] > SEQ_LEN:
                    num_seq = num_seq[:SEQ_LEN]
                else:
                    padding = num_seq[-1].unsqueeze(0).repeat(SEQ_LEN - num_seq.shape[0], 1)
                    num_seq = torch.cat((num_seq, padding), dim=0)

            if torch.isnan(num_seq).any():
                num_seq = torch.nan_to_num(num_seq, nan=0.0)

            return img_seq, num_seq, label
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}. Returning dummy sample.")
            return torch.randn(SEQ_LEN, 3, 224, 224), torch.randn(SEQ_LEN, 47), 0

def get_dataloaders(data_dir=DATA_DIR, batch_size=32, num_workers=16):
    
    # --- NEW: Load class_to_idx and create class_names list ---
    class_to_idx_path = os.path.join(data_dir, "class_to_idx.json")
    if not os.path.exists(class_to_idx_path):
        raise FileNotFoundError(f"FATAL: class_to_idx.json not found in {data_dir}")
        
    with open(class_to_idx_path, 'r') as f:
        class_to_idx = json.load(f)
    
    num_classes = len(class_to_idx)
    # Create a list of class names in the correct order
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]
    # --- END NEW ---
        
    train_dataset = YogaPoseSequentialDataset(data_dir, 'train', class_to_idx)
    valid_dataset = YogaPoseSequentialDataset(data_dir, 'valid', class_to_idx)
    test_dataset = YogaPoseSequentialDataset(data_dir, 'test', class_to_idx)
    
    max_workers = os.cpu_count() // 2
    num_workers = min(max_workers, num_workers) if max_workers > 0 else 1
    
    print(f"Dataloaders ready. Using {num_workers} workers.")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}")
    
    # --- NEW: Return class_names ---
    return train_loader, valid_loader, test_loader, num_classes, class_names

if __name__ == "__main__":
    # Test the dataloader
    print("Testing Dataloader...")
    train_loader, _, _, num_classes, class_names = get_dataloaders(batch_size=4)
    print(f"Found {num_classes} classes.")
    print(f"Class names: {class_names}")
    
    img_seq, num_seq, label = next(iter(train_loader))
    
    print(f"Batch Image Shape:   {img_seq.shape}")
    print(f"Batch Num Shape:     {num_seq.shape}")
    print(f"Batch Label Shape:   {label.shape}")
    
    assert img_seq.shape == (4, SEQ_LEN, 3, 224, 224), "Image shape mismatch!"
    assert num_seq.shape == (4, SEQ_LEN, 47), "Numerical shape mismatch!"
    
    print("\nDataloader test passed!")