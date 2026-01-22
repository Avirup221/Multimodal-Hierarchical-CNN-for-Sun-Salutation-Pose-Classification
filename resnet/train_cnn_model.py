import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import the updated modules
from models import get_model # This will get our Multimodal QuadtreeCNN or StandardResNetCNN
from dataloader import get_dataloaders, IMAGE_SIZE 

# --- Configuration (Hyperparameters & Paths) ---
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'

# --- Ablation Study Mode ---
# IMPORTANT: Change this variable for each experiment:
# 'fusion': Uses both image and numerical features (current best)
# 'image_only': Uses only image features (Quadtree CNN)
# 'numerical_only': Uses only numerical features (MLP)
# 'standard_resnet_only': Uses a standard ResNet18 for image features (new mode)
TRAINING_MODE = 'standard_resnet_only' # <<< CHANGE THIS FOR EACH EXPERIMENT

# Path where your trained model checkpoints will be saved
# The filename will now include the training mode
MODEL_SAVE_PATH = f'multimodal_quadtree_cnn_pose_model_{TRAINING_MODE}.pth' 

BATCH_SIZE = 16 
NUM_EPOCHS = 20 
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4 # L2 Regularization to combat overfitting
RANDOM_SEED = 42 

# Ensure reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for mode: {TRAINING_MODE}")

if __name__ == '__main__':
    # --- 1. Get DataLoaders ---
    print("Loading data and creating DataLoaders...")
    
    train_loader, val_loader, class_names, num_classes = get_dataloaders(
        data_root=STILL_IMAGE_DATASET_ROOT, 
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    
    print(f"Number of classes: {num_classes}, Class names: {class_names}")

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print("\nInitializing model, loss function, and optimizer...")
    # Pass the TRAINING_MODE to the get_model function
    model = get_model(num_classes=num_classes, device=device, mode=TRAINING_MODE) 

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- 3. Training Loop ---
    print("\nStarting training loop...")
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train] ({TRAINING_MODE})", leave=False)

        for batch_idx, (images, numerical_features, labels) in enumerate(train_pbar): 
            images = images.to(device)
            # numerical_features are still loaded by the DataLoader,
            # but the StandardResNetCNN's forward method will ignore them.
            numerical_features = numerical_features.to(device) 
            labels = labels.to(device)

            optimizer.zero_grad()

            # Pass both inputs to the model. The model's forward handles the mode.
            # For StandardResNetCNN, numerical_features will be ignored.
            outputs = model(images, numerical_features) 
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_pbar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / total_train
        epoch_train_accuracy = (correct_train / total_train) * 100

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val ] ({TRAINING_MODE})", leave=False)

        with torch.no_grad(): 
            for batch_idx, (images, numerical_features, labels) in enumerate(val_pbar): 
                images = images.to(device)
                numerical_features = numerical_features.to(device) 
                labels = labels.to(device)

                outputs = model(images, numerical_features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix(loss=loss.item())

        epoch_val_loss = val_loss / total_val
        epoch_val_accuracy = (correct_val / total_val) * 100

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Epoch {epoch+1} finished. Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%. "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}% (Mode: {TRAINING_MODE})")

        if epoch_val_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved: Validation accuracy improved to {best_val_accuracy:.2f}% (Mode: {TRAINING_MODE})")

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% (Mode: {TRAINING_MODE})")

    # --- 4. Plot Training History ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training & Validation Accuracy (Mode: {TRAINING_MODE})')

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training & Validation Loss (Mode: {TRAINING_MODE})')
    plt.tight_layout()
    plt.show()

