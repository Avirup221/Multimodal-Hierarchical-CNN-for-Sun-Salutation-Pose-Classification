import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    r2_score
)

from model import get_model
from dataloader import get_dataloaders, IMAGE_SIZE

# ----------------- CONFIG -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

STILL_IMAGE_DATASET_ROOT = r"E:/User/my work/Summer project/Code/flat_image_dataset_final"
BATCH_SIZE = 16

# 🔴 EDIT THIS: paths to your saved .pth files
MODEL_PATHS = {
    "quadtree":      r"E:\User\my work\Summer project\Code\Model_Training\comparative analysis\multimodal_quadtree_pose_model.pth",
    "resnet18":      r"E:\User\my work\Summer project\Code\Model_Training\comparative analysis\multimodal_resnet18_pose_model.pth",
    "vgg16":         r"E:\User\my work\Summer project\Code\Model_Training\comparative analysis\multimodal_vgg16_pose_model.pth",
    "mobilenet_v2":  r"E:\User\my work\Summer project\Code\Model_Training\comparative analysis\multimodal_mobilenet_v2_pose_model.pth"
}
# -------------------------------------------


def plot_confusion_matrix(cm, class_names, title):
    """Pure Matplotlib confusion matrix (no seaborn)."""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, val_loader, class_names, model_name):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, numerical_features, labels in val_loader:
            images = images.to(device)
            numerical_features = numerical_features.to(device)
            labels = labels.to(device)

            outputs = model(images, numerical_features)
            _, predicted = torch.max(outputs.data, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # ---- Metrics ----
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )
    # Not usually used for classification, but computing as you asked:
    r2 = r2_score(all_labels, all_preds)

    print(f"\n===== {model_name.upper()} (Validation) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall   : {recall:.4f} (weighted)")
    print(f"F1-score : {f1:.4f} (weighted)")
    print(f"R² score : {r2:.4f}  (NOTE: not standard for classification)")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, f"Confusion Matrix - {model_name.upper()}")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "r2": r2,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    # 1. Load data
    print("Loading validation data...")
    train_loader, val_loader, class_names, num_classes = get_dataloaders(
        data_root=STILL_IMAGE_DATASET_ROOT,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    print(f"Classes ({num_classes}): {class_names}")

    all_results = {}
    model_names_in_order = list(MODEL_PATHS.keys())

    # 2. Loop over each saved model
    for model_name, ckpt_path in MODEL_PATHS.items():
        print(f"\n---------------------------")
        print(f" Evaluating model: {model_name}")
        print(f" Checkpoint: {ckpt_path}")
        print(f"---------------------------")

        # Build same architecture as during training
        model = get_model(model_name=model_name, num_classes=num_classes, device=device)

        # Load saved weights
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Evaluate & store metrics
        metrics = evaluate_model(model, val_loader, class_names, model_name)
        all_results[model_name] = metrics

    # 3. Optional: Plot comparison of metrics across models
    print("\nPlotting metric comparison across models...")

    metric_names = ["accuracy", "precision", "recall", "f1"]
    for metric in metric_names:
        plt.figure()
        values = [all_results[m][metric] for m in model_names_in_order]
        plt.bar(model_names_in_order, values)
        plt.ylim(0, 1)
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} comparison across models")
        plt.tight_layout()
        plt.show()

    # R² comparison (separate)
    plt.figure()
    r2_values = [all_results[m]["r2"] for m in model_names_in_order]
    plt.bar(model_names_in_order, r2_values)
    plt.ylabel("R² score")
    plt.title("R² comparison across models")
    plt.tight_layout()
    plt.show()

    print("\nDone evaluating all models.")
