import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import the model definition and dataloader
from models import get_model
from dataloader import get_dataloaders, IMAGE_SIZE

# --- Configuration ---
MODEL_SAVE_PATH = 'quadtree_pose_model.pth'
STILL_IMAGE_DATASET_ROOT = r'E:/User/my work/Summer project/Code/flat_image_dataset_final'

BATCH_SIZE = 16 
NUM_EPOCHS = 10 
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42 
EARLY_STOPPING_PATIENCE = 5

# --- Setup ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if __name__ == '__main__':
    # --- DataLoaders ---
    train_loader, val_loader, class_names, num_classes = get_dataloaders(
        data_root=STILL_IMAGE_DATASET_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    print(f"Found {num_classes} classes.")

    # --- Model, Loss, Optimizer ---
    print("\nInitializing QuadtreeCNN model...")
    # The get_model function from your file correctly loads the QuadtreeCNN
    model = get_model(num_classes=num_classes, device=device,model_name = "quadtree")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    print("\nStarting training...")
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(NUM_EPOCHS):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
        for images, numerical_features, labels in train_pbar: 
            images, numerical_features, labels = images.to(device), numerical_features.to(device), labels.to(device)
            optimizer.zero_grad()
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

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, numerical_features, labels in val_loader:
                images, numerical_features, labels = images.to(device), numerical_features.to(device), labels.to(device)
                outputs = model(images, numerical_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        epoch_val_loss = val_loss / total_val
        epoch_val_accuracy = (correct_val / total_val) * 100
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Epoch {epoch+1} | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}%")

        # Early Stopping Logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Model saved to {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs.")
            break

    print("\nTraining complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.suptitle('Training History for QuadtreeCNN', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=len(history['val_loss']) - 1 - epochs_no_improve, color='r', linestyle='--', label='Best Model Epoch')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()