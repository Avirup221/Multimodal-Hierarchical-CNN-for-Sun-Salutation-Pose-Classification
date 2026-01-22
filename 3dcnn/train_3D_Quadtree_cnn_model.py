import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix 
import seaborn as sns 

# Import the updated modules
from models import get_model 
from dataloaders import get_dataloaders, IMAGE_SIZE, NUM_NUMERICAL_FEATURES, NUM_CLASSES 

# --- Configuration (Hyperparameters & Paths) ---
# This data_root is primarily for 2D modes, ignored for 3D sequential modes
DATASET_ROOT_FOR_2D_MODES = r'/home/avirupd/summer_project/flat_image_dataset_final'

# IMPORTANT: Set to 'hybrid_quadtree_3d_image_only' for the new hybrid model (image-only)
TRAINING_MODE = 'hybrid_quadtree_3d_image_only' 

MODEL_SAVE_PATH = f'multimodal_quadtree_cnn_pose_model_{TRAINING_MODE}.pth' 

PLOTS_SAVE_DIR = r'/home/avirupd/summer_project/training_plots'

BATCH_SIZE = 8 
NUM_EPOCHS = 50 
LEARNING_RATE = 0.00005 # Keep consistent with other 3D models initially
WEIGHT_DECAY = 5e-4 
RANDOM_SEED = 42 

SEQUENCE_LENGTH = 5 # Must match the sequence length of your prepared data

GRAD_CLIP_NORM = 1.0 

LR_SCHEDULER_FACTOR = 0.5 
LR_SCHEDULER_PATIENCE = 5 

EARLY_STOPPING_PATIENCE = 10 
MIN_DELTA = 0.001 

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
    # Create directory for plots if it doesn't exist
    os.makedirs(PLOTS_SAVE_DIR, exist_ok=True)

    # --- 1. Get DataLoaders ---
    print("Loading data and creating DataLoaders...")
    
    # get_dataloaders will use YogaPoseSequenceDataset for 3D modes
    train_loader, val_loader, class_to_idx, num_classes = get_dataloaders(
        data_root=DATASET_ROOT_FOR_2D_MODES, # This is ignored for 3D sequential modes
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_numerical_features=NUM_NUMERICAL_FEATURES,
        mode=TRAINING_MODE # This will trigger YogaPoseSequenceDataset
    )
    
    print(f"Number of classes: {num_classes}, Class names: {class_to_idx}")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)] 

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print("\nInitializing model, loss function, and optimizer...")
    # get_model will instantiate HybridQuadtree3DCNN for 'hybrid_quadtree_3d_image_only' mode
    model = get_model(
        num_classes=num_classes, 
        device=device, 
        mode=TRAINING_MODE, 
        numerical_feature_dim=NUM_NUMERICAL_FEATURES, # This is ignored by HybridQuadtree3DCNN in image_only mode
        sequence_length=SEQUENCE_LENGTH 
    ) 

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, 
                                  patience=LR_SCHEDULER_PATIENCE, min_lr=1e-7)

    # --- 3. Training Loop ---
    print("\nStarting training loop...")
    best_val_accuracy = 0.0
    best_val_loss = float('inf') 
    epochs_no_improve = 0 
    
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(NUM_EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train] ({TRAINING_MODE})", leave=False)

        for batch_idx, (images_or_sequences, numerical_features_or_sequences, labels) in enumerate(train_pbar): 
            images_or_sequences = images_or_sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # For HybridQuadtree3DCNN image_only, numerical_features_or_sequences are ignored by forward method
            outputs = model(images_or_sequences, numerical_features_or_sequences) 
            loss = criterion(outputs, labels)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()

            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in training batch {batch_idx}, epoch {epoch+1}. Skipping loss accumulation for this batch.")
                continue 

            running_loss += loss.item() * labels.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_pbar.set_postfix(loss=loss.item())

        if total_train == 0:
            print(f"Epoch {epoch+1} finished. No valid training samples processed due to NaN loss. Skipping loss/accuracy calculation.")
            epoch_train_loss = float('nan')
            epoch_train_accuracy = float('nan')
        else:
            epoch_train_loss = running_loss / total_train
            epoch_train_accuracy = (correct_train / total_train) * 100

            history['train_loss'].append(epoch_train_loss)
            history['train_accuracy'].append(epoch_train_accuracy)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        all_val_labels = [] 
        all_val_predictions = [] 

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val ] ({TRAINING_MODE})", leave=False)

        with torch.no_grad(): 
            for batch_idx, (images_or_sequences, numerical_features_or_sequences, labels) in enumerate(val_pbar): 
                images_or_sequences = images_or_sequences.to(device)
                labels = labels.to(device)

                outputs = model(images_or_sequences, numerical_features_or_sequences)
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected in validation batch {batch_idx}, epoch {epoch+1}. Skipping loss accumulation for this batch.")
                    continue 

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                val_pbar.set_postfix(loss=loss.item())

                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(predicted.cpu().numpy())
        
        if total_val == 0:
            print(f"Epoch {epoch+1} finished. No valid validation samples processed due to NaN loss. Skipping loss/accuracy calculation.")
            epoch_val_loss = float('nan')
            epoch_val_accuracy = float('nan')
        else:
            epoch_val_loss = val_loss / total_val
            epoch_val_accuracy = (correct_val / total_val) * 100

        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)

        print(f"Epoch {epoch+1} finished. Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%. "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.2f}% (Mode: {TRAINING_MODE})")

        scheduler.step(epoch_val_loss)

        if not np.isnan(epoch_val_loss): 
            if epoch_val_loss < best_val_loss - MIN_DELTA: 
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"✅ Model saved: Validation loss improved to {best_val_loss:.4f} (Mode: {TRAINING_MODE})")
                if not np.isnan(epoch_val_accuracy) and epoch_val_accuracy > best_val_accuracy:
                    best_val_accuracy = epoch_val_accuracy
            else:
                epochs_no_improve += 1
                print(f"Validation loss did not improve for {epochs_no_improve} epochs.")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} epochs without improvement.")
                    break 

    print("\nTraining complete!")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% (Mode: {TRAINING_MODE})")
    print(f"Best Validation Loss: {best_val_loss:.4f} (Mode: {TRAINING_MODE})")

    # --- 4. Plot Training History and Save ---
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    accuracy_plot_filename = os.path.join(PLOTS_SAVE_DIR, f'accuracy_plot_{TRAINING_MODE}_{timestamp}.png')
    loss_plot_filename = os.path.join(PLOTS_SAVE_DIR, f'loss_plot_{TRAINING_MODE}_{timestamp}.png')
    confusion_matrix_filename = os.path.join(PLOTS_SAVE_DIR, f'confusion_matrix_{TRAINING_MODE}_{timestamp}.png') 

    plt.figure(figsize=(10, 6)) 
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training & Validation Accuracy (Mode: {TRAINING_MODE})')
    plt.legend()
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig(accuracy_plot_filename) 
    plt.close() 

    plt.figure(figsize=(10, 6)) 
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training & Validation Loss (Mode: {TRAINING_MODE})')
    plt.legend()
    plt.grid(True) 
    plt.tight_layout()
    plt.savefig(loss_plot_filename) 
    plt.close() 

    if all_val_labels and all_val_predictions: 
        cm = confusion_matrix(all_val_labels, all_val_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix (Validation Set - Mode: {TRAINING_MODE})')
        plt.tight_layout()
        plt.savefig(confusion_matrix_filename)
        plt.close()
        print(f"Confusion matrix saved to: {confusion_matrix_filename}")
    else:
        print("No validation data collected for confusion matrix. Skipping plot.")

    print(f"Training plots saved to: {PLOTS_SAVE_DIR}")

