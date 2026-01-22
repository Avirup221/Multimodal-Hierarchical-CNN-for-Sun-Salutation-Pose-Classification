import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import json
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

# --- 1. Import from our new, correct files ---
from fact_model import FactModel
from fact_dataloader import get_dataloaders, SEQ_LEN

# --- 2. Configuration ---
DATA_DIR = "/home/avirupd/summer_project/augmented_sequential_dataset_seq4"
MODEL_SAVE_DIR = 'saved_models'
MODEL_NAME = f"fact_model_seq{SEQ_LEN}.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 3. Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 16

RANDOM_SEED = 42

# --- Setup ---
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

# --- Training & Validation Functions (Unchanged) ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    pbar = tqdm(dataloader, desc=f"Epoch [Train]", leave=False)
    
    for img_seq, num_seq, labels in pbar:
        img_seq, num_seq, labels = img_seq.to(device), num_seq.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(img_seq, num_seq)
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss):
            print(f"Warning: NaN loss detected in training. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_train / total_train) * 100:.2f}%")
        
    epoch_loss = running_loss / total_train if total_train > 0 else 0
    epoch_acc = (correct_train / total_train) * 100 if total_train > 0 else 0
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0
    pbar = tqdm(dataloader, desc=f"Epoch [Valid]", leave=False)
    
    with torch.no_grad():
        for img_seq, num_seq, labels in pbar:
            img_seq, num_seq, labels = img_seq.to(device), num_seq.to(device), labels.to(device)
            outputs = model(img_seq, num_seq)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in validation. Skipping batch.")
                continue

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_val / total_val) * 100:.2f}%")
            
    epoch_loss = running_loss / total_val if total_val > 0 else 0
    epoch_acc = (correct_val / total_val) * 100 if total_val > 0 else 0
    return epoch_loss, epoch_acc

# --- NEW: Plotting and Evaluation Functions ---
def plot_training_history(history, output_filename="training_history.png"):
    """Saves a plot of training/validation loss and accuracy."""
    print(f"\nSaving training history plot to {output_filename}...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot Accuracy
    ax1.plot(history['train_accuracy'], label='Train Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title(f'Model Accuracy (Best Val: {max(history["val_accuracy"]):.2f}%)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True, linestyle='--')
    
    # Plot Loss
    ax2.plot(history['train_loss'], label='Train Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title(f'Model Loss (Min Val: {min(history["val_loss"]):.4f})')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print("Plot saved.")

def evaluate_and_plot_cm(model, loader, device, class_names, output_filename="confusion_matrix.png"):
    """Runs evaluation on the test set and saves a confusion matrix plot."""
    print(f"\nEvaluating on test set and generating confusion matrix...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for img_seq, num_seq, labels in tqdm(loader, desc="Test Set"):
            img_seq, num_seq, labels = img_seq.to(device), num_seq.to(device), labels.to(device)
            
            outputs = model(img_seq, num_seq)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate accuracy
    test_acc = (np.array(all_preds) == np.array(all_labels)).sum() / len(all_labels)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues')
    plt.title(f'Confusion Matrix (Test Acc: {test_acc*100:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(output_filename)
    print(f"Confusion matrix saved to {output_filename}")
    return test_acc

# --- Main Training ---
def main():
    # --- 1. Get DataLoaders ---
    # This now returns class_names
    train_loader, valid_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Training model for {num_classes} classes.")
    
    # --- 2. Initialize Model, Loss, and Optimizer ---
    model = FactModel(
        num_classes=num_classes,
        seq_len=SEQ_LEN 
    ).to(device)
    
    print(f"Model loaded. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} Million")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)
    
    # --- 3. Training Loop ---
    print("\n--- Starting Training ---")
    best_val_accuracy = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, valid_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.2f}%"
        )
        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"    -> ✅ New best model saved to {MODEL_SAVE_PATH} (Val Acc: {best_val_accuracy:.2f}%)")

    print("\n--- Training Complete! ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    
    # --- 4. Plot History ---
    plot_training_history(history, output_filename=f"training_history_seq{SEQ_LEN}.png")
    
    # --- 5. Final Test with Best Model ---
    print("\n--- Loading best model for final test ---")
    # Load the best performing model
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    except Exception as e:
        print(f"Error loading best model: {e}. Running test with final epoch model.")

    evaluate_and_plot_cm(
        model, 
        test_loader, 
        device, 
        class_names, 
        output_filename=f"confusion_matrix_seq{SEQ_LEN}.png"
    )

if __name__ == '__main__':
    main()