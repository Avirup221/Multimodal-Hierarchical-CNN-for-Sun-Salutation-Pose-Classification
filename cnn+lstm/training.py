import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

# Import from our files
from dataloader import get_dataloaders # Reuse the dataloader we made for FACT
from models import get_model # Import the new models

# --- Configuration ---
# --- THIS IS THE CRITICAL CONFIGURATION ---
MODEL_TYPE = 'cnn_lstm' # Options: 'cnn_lstm', '3d_cnn'
DATA_DIR = "/home/avirupd/summer_project/new_sequential_dataset_seq4_v3"
SEQ_LEN = 4
# ----------------------------------------

MODEL_SAVE_DIR = 'saved_models'
MODEL_NAME = f"{MODEL_TYPE}_seq{SEQ_LEN}.pth"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Hyperparameters ---
BATCH_SIZE = 32
NUM_EPOCHS = 50 
LEARNING_RATE = 1e-4
NUM_WORKERS = 16

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for img_seq, num_seq, labels in tqdm(dataloader, desc="[Train]", leave=False):
        img_seq, num_seq, labels = img_seq.to(device), num_seq.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(img_seq, num_seq) # Pass both inputs
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / total, (correct / total) * 100

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for img_seq, num_seq, labels in tqdm(dataloader, desc="[Valid]", leave=False):
            img_seq, num_seq, labels = img_seq.to(device), num_seq.to(device), labels.to(device)
            
            outputs = model(img_seq, num_seq)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return running_loss / total, (correct / total) * 100

def main():
    # 1. Get Dataloaders
    train_loader, valid_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    print(f"Training {MODEL_TYPE} for {num_classes} classes.")

    # 2. Initialize Model
    model = get_model(MODEL_TYPE, num_classes, device, seq_len=SEQ_LEN)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Loaded: {MODEL_TYPE.upper()} | Trainable Params: {num_params/1e6:.2f}M")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    # 3. Training Loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n--- Starting Training ---")
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, valid_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {time.time()-start:.0f}s | "
              f"Train Acc: {train_acc:.2f}% | Valid Acc: {val_acc:.2f}%")
        
        history['train_acc'].append(train_acc); history['val_acc'].append(val_acc)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✅ Best model saved! ({val_acc:.2f}%)")

    print(f"\n--- Training Complete. Best Val Acc: {best_val_acc:.2f}% ---")
    
    # 4. Plot History
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{MODEL_TYPE.upper()} Accuracy')
    plt.legend()
    plt.savefig(f'{MODEL_TYPE}_history.png')
    print("History plot saved.")

if __name__ == "__main__":
    main()