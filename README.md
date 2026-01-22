# SURYA: Spatiotemporal Understanding of Recognition for Yoga Asanas  
### (Sun Salutation / Surya Namaskar Pose Recognition)

A **multimodal spatiotemporal deep learning framework** to recognize **Surya Namaskar yoga poses** using **spatial + temporal understanding**, combining:

- ✅ RGB image-based visual features (CNN / ViT)
- ✅ Pose geometry features (**47 engineered MediaPipe features**)
- ✅ Multiscale regional learning using **QuadtreeCNN**
- ✅ Temporal sequence modeling (**CNN+LSTM, 3D CNN, ViT**)
- ✅ Viewpoint robustness via generative multi-view augmentation (**Zero123-Plus diffusion**)

> 📌 Project based on report: *BDA Spatiotemporal Framework for Sun Salutation Pose Recognition*  
> **Author:** Avirup Das (B2430041)  
> Dept. of Computer Science, RKMVERI

---

## ✨ Demo / Goal
This project aims to automatically recognize Surya Namaskar poses and transitions, enabling:
- feedback systems for yoga practitioners
- posture correctness monitoring
- spatiotemporal understanding of human movement

---

## 📌 Key Contributions
### ✅ Spatial Understanding (Per-frame)
1. Comparison of CNN backbones:
   - VGG-16
   - MobileNet-V2
   - ResNet-18 ✅ (selected)
2. Proposed **QuadtreeCNN**
   - hierarchical quadrant splitting of feature maps
   - learns fine-grained local regions (arms, legs, torso zones)
3. **Multimodal fusion**
   - ResNet features + Quadtree multiscale features
   - fused with **47 pose-engineered features**
4. Interpretability via **Grad-CAM**

### ✅ Temporal Understanding (Sequences)
1. CNN + LSTM multimodal sequence classifier
2. 3D CNN-based spatiotemporal modeling (3D Quadtree)
3. Transformer/Vision Transformer (ViT) based spatiotemporal fusion

### ✅ Augmentation & Robustness
- background removal with **rembg**
- multi-view diffusion generation using **Zero123-Plus**
- grid slicing into multiple viewpoints
- sliding-window sequence generation

---

## 📂 Dataset
### Dataset Source
- Dataset name: `surya_namaskar.v4i.coco`
- Frames extracted: **~3,500 frames from 5 videos**
- Structure: train / valid / test split

---

## 🔢 Engineered Pose Features (47 Features)
Pose features are extracted using **MediaPipe Pose** and saved per image as `.npy` files.

Includes:
- 33 landmark visibility scores
- joint angles: elbows, shoulders, hips, knees
- normalized distances (wrist-wrist, ankle-ankle, etc.)
- torso inclination and alignment
- torso variance ratio

---

## 🗂️ Final Dataset Structure
```bash
flat_image_dataset_final/
│── train/
│── valid/
│── test/
    ├── <pose_class_name>/
        ├── image_XXXX.jpg
        ├── image_XXXX.npy
---

## ✅ Additional Generated Files

During preprocessing and feature engineering, the following statistics files are generated:

- `class_feature_means.json`
- `class_feature_stds.json`

---

## 🧠 Model Architecture

### 1) Spatial Pipeline (Multimodal)

**Flow:**

Image → **ResNet-18 backbone** → feature map  
→ **Quadtree splitting + global feature branch**  
→ **fusion with 47 pose features**  
→ classifier

---

### 2) QuadtreeCNN (Multiscale Spatial Learning)

Feature maps extracted from **ResNet layer (`layer3`)** are split into hierarchical quadrants:

- **Level-1:** `2×2` quadrants (4 blocks)
- **Level-2:** optional deeper split for fine-grained regions

Each region contributes an embedding and is fused with:

- global embedding  
- pose numeric embedding (**47 engineered features**)  

---

### 3) Temporal Pipeline

Sequences are constructed as:

- multiple frames per clip
- multiple generated viewpoints
- sliding window over frames

Models used:

- **CNN + LSTM** (baseline sequence modeling)
- **3D CNN** (learns motion as volume)
- **ViT spatiotemporal fusion**

---

## 📊 Results

### ✅ Spatial Model Performance (Test Set)

| Model | Top-1 Accuracy | Precision | Recall | F1 |
|------|---------------:|----------:|-------:|----:|
| VGG-16 | 95.8% | 0.9679 | 0.9623 | 0.9622 |
| MobileNet-V2 | 96.23% | 0.9645 | 0.9623 | 0.9627 |
| ResNet-18 | 96.78% | 0.9623 | 0.9627 | 0.9622 |
| ✅ QuadtreeCNN (ResNet-18 + Fusion) | **97.07%** | **0.9728** | **0.9707** | **0.9708** |

---

### ✅ Temporal Model Observations

- **CNN+LSTM achieved ~71.68% validation accuracy**
- **3D Quadtree CNN showed improved stability and overall recognition quality**
- Some classes had weak recall due to low validation support

---

## 🔍 Interpretability (Grad-CAM)

Grad-CAM visualization confirms:

- attention on **shoulders/arms** during raised poses
- attention on **hips/knees** during lunge-based poses
- quadtree regions focus on correct body parts (**regional explainability**)

---

## 🧪 Temporal Dataset Preparation Pipeline

### Stage 1 — Background Removal
- tool: `rembg`
- improves focus and diffusion consistency

### Stage 2 — Multi-view Augmentation
- tool: **Zero123-Plus diffusion model**
- generates multi-view images from single image

### Stage 3 — Grid Slicing
- output grid: **3×2**
- sliced into **6 viewpoints**

### Stage 4 — Sequence Construction
- sequences built clip-wise & viewpoint-wise
- sliding window over frames:
  - window size `T`
  - stride `S`

---

## 🛠️ Environment Setup

Because diffusion augmentation + model training may conflict in dependencies, use **multi-env workflow**:

### Recommended Environments

- `rembg_env` → background removal  
- `genai_env` → Zero123-Plus multi-view generation  
- `train_env` → training spatiotemporal models  

---

## ⚙️ Installation

### Clone Repo
```bash
git clone https://github.com/<your-username>/SURYA-Yoga-Pose-Recognition.git
cd SURYA-Yoga-Pose-Recognition
