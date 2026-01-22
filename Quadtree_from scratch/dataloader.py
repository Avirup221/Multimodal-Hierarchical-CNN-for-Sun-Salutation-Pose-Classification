import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import json

IMAGE_SIZE = (224, 224) 
NUM_NUMERICAL_FEATURES = 47 

STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'
CLASS_FEATURE_MEANS_FILE = os.path.join(STILL_IMAGE_DATASET_ROOT, 'class_feature_means.json')

class YogaPoseFrameDataset(Dataset):
    def __init__(self, data_dir, img_size, is_train=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train

        self.image_paths = []
        self.feature_paths = [] 
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []
        self.class_feature_means = None 

        if self.is_train:
            self.image_transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)), 
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
                transforms.RandomRotation(degrees=10), 
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else: 
            self.image_transform = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        self._load_data()
        self._load_class_feature_means() 

    def _load_data(self):
        all_class_names = sorted([d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))])
        self.idx_to_class = all_class_names 
        self.class_to_idx = {name: i for i, name in enumerate(self.idx_to_class)}

        for class_name in self.idx_to_class:
            class_path = os.path.join(self.data_dir, class_name) 
            for filename in sorted(os.listdir(class_path)):
                if filename.endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_path, filename)
                    feature_path = os.path.join(class_path, os.path.splitext(filename)[0] + '.npy')
                    
                    if os.path.exists(feature_path):
                        self.image_paths.append(img_path)
                        self.feature_paths.append(feature_path)
                        self.labels.append(self.class_to_idx[class_name])

    def _load_class_feature_means(self):
        with open(CLASS_FEATURE_MEANS_FILE, 'r') as f:
            self.class_feature_means = json.load(f)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        feature_path = self.feature_paths[idx]
        label_int = self.labels[idx]
        label_string = self.idx_to_class[label_int] 

        img = Image.open(img_path).convert('RGB')
        image_tensor = self.image_transform(img)

        numerical_features_np = np.load(feature_path)
        numerical_features_tensor = torch.tensor(numerical_features_np, dtype=torch.float32)

        if self.class_feature_means is not None:
            feature_names = list(self.class_feature_means[label_string].keys())
            for i, feature_name in enumerate(feature_names):
                if torch.isnan(numerical_features_tensor[i]):
                    numerical_features_tensor[i] = self.class_feature_means[label_string].get(feature_name, 0.0) 

        label_tensor = torch.tensor(label_int, dtype=torch.long)
        return image_tensor, numerical_features_tensor, label_tensor

def get_dataloaders(data_root, batch_size, image_size):
    train_data_path = os.path.join(data_root, 'train')
    val_data_path = os.path.join(data_root, 'valid') 

    train_dataset_obj = YogaPoseFrameDataset(train_data_path, image_size, is_train=True)
    train_loader = DataLoader(train_dataset_obj, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
    
    val_dataset_obj = YogaPoseFrameDataset(val_data_path, image_size, is_train=False)
    val_loader = DataLoader(val_dataset_obj, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)

    class_names = train_dataset_obj.idx_to_class
    num_classes = len(class_names)

    return train_loader, val_loader, class_names, num_classes