# 🛰️ SentinalVision

**Off-Road Autonomous Navigation System**

An end-to-end deep learning pipeline for safe off-road path planning using terrain segmentation, cost heatmap generation, and A* pathfinding.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Training](#-training)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

SentinalVision enables autonomous vehicles, drones, and robots to navigate safely through challenging off-road terrains like forests, deserts, and disaster zones. The system addresses key challenges in unstructured environments:

- **No lane markings** or traditional road infrastructure
- **Unpredictable terrain** with varying traversability
- **Safety-critical path planning** avoiding rocks, logs, and dangerous areas

### Pipeline Flow

```
Image → DINOv2 Segmentation → Cost Heatmap → A* Pathfinding → Safe Route
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🧠 **DINOv2 Backbone** | Pre-trained ViT-Base foundation model (768 dim, 86M params) |
| 🎨 **10-Class Segmentation** | Classifies terrain into traversability categories |
| 🗺️ **Cost Heatmap** | Converts segmentation to continuous traversal difficulty map |
| 🛤️ **A* Pathfinding** | 8-directional movement with terrain-aware cost function |
| 🌐 **Gradio Web UI** | Interactive point selection and real-time visualization |
| 📊 **Batch Processing** | Process multiple images with comparative analysis |

---

## 🏗️ Architecture

### Model Architecture

| Component | Technology |
|-----------|------------|
| Backbone | DINOv2-ViT-Base (`dinov2_vitb14`) - 768 dim embedding |
| Segmentation Head | Custom UNet decoder with Attention blocks (4 stages) |
| Loss Function | Combined: 0.6×Focal + 0.4×Dice Loss |
| Pathfinding | A* Algorithm with 8-directional movement |
| Frontend | Gradio (Python-based web UI) |
| Framework | PyTorch 2.0+ |

### Terrain Classes & Traversal Costs

| Class | Cost | Risk Level |
|-------|------|------------|
| Background | 1 | ✅ Very Safe |
| Trees | 3 | ✅ Safe |
| Lush Bushes | 4 | ✅ Safe |
| Dry Grass | 2 | ✅ Very Safe |
| Dry Bushes | 5 | ⚠️ Moderate |
| Ground Clutter | 6 | ⚠️ Caution |
| Logs | 15 | ⚠️ Warning |
| **Rocks** | **50** | 🚫 **DANGER** |
| Landscape | 1 | ✅ Very Safe |
| Sky | 0 | ❌ Impassable |

---

## 🛠️ Installation

### Prerequisites

- **Python 3.10+**
- **CUDA 11.8+** (for GPU acceleration, optional but recommended)
- **Conda** (recommended) or pip

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/SentinalVision.git
cd SentinalVision

# Create conda environment
conda create -n sentinalvision python=3.10 -y
conda activate sentinalvision

# Install PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/SentinalVision.git
cd SentinalVision

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Windows Quick Setup

Use the provided batch scripts:

```batch
cd modal\Offroad_Segmentation_Scripts\ENV_SETUP
create_env.bat
install_packages.bat
```

---

## 🚀 Quick Start

### 1. Run the Web Interface

```bash
python frontend/app.py
```

Open your browser to: **http://127.0.0.1:7860**

### 2. Demo Instructions

1. **Upload** an off-road image
2. Click **"Run Segmentation"** to analyze terrain
3. **Adjust** start/end point sliders (live preview updates)
4. Click **"Find Safest Path"** to compute optimal route
5. **View** path on Original, Safe Areas, and Heatmap displays

---

## 📁 Project Structure

```
SentinalVision/
├── frontend/
│   └── app.py                    # Gradio web interface
│
├── modal/
│   └── Offroad_Segmentation_Scripts/
│       ├── train_segmentation.py  # Training script
│       ├── test_segmentation.py   # Testing/inference script
│       ├── run_single.py          # Single image inference
│       ├── visualize.py           # Visualization utilities
│       ├── segmentation_head.pth  # Pre-trained model weights (12MB)
│       ├── train_stats/           # Training metrics & plots
│       └── ENV_SETUP/             # Environment setup scripts
│           ├── create_env.bat
│           ├── install_packages.bat
│           └── setup_env.bat
│
├── planner/
│   ├── mask_to_heatmap.py         # Cost map conversion
│   ├── visualize_path.py          # Path visualization
│   ├── batch_process.py           # Batch inference pipeline
│   └── compare_stats.py           # Model comparison utilities
│
├── data/
│   ├── Offroad_Segmentation_testImages/
│   │   ├── Color_Images/          # Test input images
│   │   └── Segmentation/          # Ground truth masks
│   │
│   └── Offroad_Segmentation_Training_Dataset/
│       ├── train/
│       │   ├── Color_Images/      # Training images
│       │   └── Segmentation/      # Training masks
│       └── val/
│           ├── Color_Images/      # Validation images
│           └── Segmentation/      # Validation masks
│
├── outputs/                       # Generated outputs
│   ├── mask.png                   # Segmentation output
│   ├── heatmap.npy                # Cost heatmap (numpy)
│   ├── heatmap_vis.png            # Heatmap visualization
│   ├── safe_areas.png             # Safe traversal overlay
│   └── comparison.png             # Side-by-side results
│
├── requirements.txt               # Python dependencies
├── folder_structure.txt           # Project structure reference
└── README.md                      # This file
```

---

## ⚙️ Configuration

### Environment Variables

No environment variables are required. All paths are resolved relative to the script locations.

### Key Configuration Parameters

Located in `frontend/app.py` and `modal/Offroad_Segmentation_Scripts/train_segmentation.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W` | 476 | Input width (divisible by 14 for ViT patches) |
| `H` | 266 | Input height (divisible by 14 for ViT patches) |
| `n_classes` | 10 | Number of terrain classes |
| `batch_size` | 4 | Training batch size |
| `lr` | 5e-4 | Initial learning rate |
| `n_epochs` | 50 | Training epochs |
| `warmup_epochs` | 5 | Learning rate warmup period |
| `grad_clip` | 1.0 | Gradient clipping value |

### Model Paths

| Path | Description |
|------|-------------|
| `modal/Offroad_Segmentation_Scripts/segmentation_head.pth` | Pre-trained segmentation head weights |

### Cost Map Configuration

Modify the `COST_MAP` dictionary in `frontend/app.py` or `planner/mask_to_heatmap.py`:

```python
COST_MAP = {
    0: 1,     # Background
    1: 3,     # Trees
    2: 4,     # Lush Bushes
    3: 2,     # Dry Grass
    4: 5,     # Dry Bushes
    5: 6,     # Ground Clutter
    6: 15,    # Logs
    7: 50,    # Rocks (highest danger)
    8: 1,     # Landscape
    9: 0,     # Sky (impassable)
}
```

---

## 🏋️ Training

### Train from Scratch

```bash
cd modal/Offroad_Segmentation_Scripts
python train_segmentation.py
```

### Training Configuration

Edit `train_segmentation.py` to adjust:

```python
# Hyperparameters
batch_size = 4          # Increase if GPU memory allows
lr = 5e-4               # Initial learning rate
n_epochs = 50           # Training epochs
warmup_epochs = 5       # LR warmup period
grad_clip = 1.0         # Gradient clipping

# Loss weights
focal_weight = 0.6      # Focal loss contribution
dice_weight = 0.4       # Dice loss contribution

# Class weights (for imbalanced classes)
class_weights = [0.3, 1.2, 1.5, 1.3, 1.8, 1.8, 2.5, 2.0, 0.8, 0.3]
```

### Data Augmentation

The training pipeline includes:

- Horizontal flip (50% probability)
- Vertical flip (20% probability)
- Random rotation (±15°)
- Random scale/crop (0.8x - 1.0x)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur (10% probability)

### Training Outputs

Generated in `modal/Offroad_Segmentation_Scripts/train_stats/`:

- `training_curves.png` - Loss and accuracy plots
- `iou_curves.png` - IoU progression
- `dice_curves.png` - Dice score progression
- `all_metrics_curves.png` - Combined metrics
- `evaluation_metrics.txt` - Detailed metrics log

---

## 📖 Usage

### Web Interface

```bash
python frontend/app.py
```

### Single Image Inference

```bash
cd modal/Offroad_Segmentation_Scripts
python run_single.py --image path/to/image.jpg
```

### Batch Processing

```bash
cd planner
python batch_process.py
```

### Generate Heatmap from Mask

```bash
cd planner
python mask_to_heatmap.py
```

### Visualize Path

```bash
cd planner
python visualize_path.py
```

---

## 📚 API Reference

### Core Functions

#### `run_segmentation(image)`
Runs the full segmentation pipeline on an input image.

**Parameters:**
- `image` (np.ndarray): Input RGB image

**Returns:**
- `mask_color` (np.ndarray): Colored segmentation mask
- `heatmap_visual` (np.ndarray): Heatmap with legend
- `safe_overlay` (np.ndarray): Safe area visualization
- `preview` (np.ndarray): Preview for point selection
- `status` (str): Status message

#### `astar(heatmap, start, end)`
A* pathfinding algorithm.

**Parameters:**
- `heatmap` (np.ndarray): Cost map
- `start` (tuple): Start coordinates (x, y)
- `end` (tuple): End coordinates (x, y)

**Returns:**
- `path` (list): List of (row, col) tuples representing the path

#### `mask_to_heatmap(mask)`
Converts segmentation mask to cost heatmap.

**Parameters:**
- `mask` (np.ndarray): Segmentation mask with class IDs (0-9)

**Returns:**
- `heatmap` (np.ndarray): Float32 array with traversal costs

---

## 📊 Performance

### Model Comparison: 10 Epochs vs 50 Epochs

| Metric | 10 Epochs | 50 Epochs | Improvement |
|--------|-----------|-----------|-------------|
| Validation IoU | 0.2959 | 0.5046 | **+70.5%** |
| Validation Dice | ~0.50 | 0.7027 | **+40.5%** |
| Validation Accuracy | 70.31% | 81.25% | **+15.6%** |
| Validation Loss | 0.5461 | 0.3605 | **-34.0%** |

### Final Training Stats (50 Epochs)

| Metric | Train | Validation |
|--------|-------|------------|
| Loss | 0.3044 | 0.3605 |
| IoU | 0.5559 | 0.5043 |
| Dice | 0.6930 | 0.7026 |
| Accuracy | 81.02% | 81.25% |

### Best Results Achieved

- **Best Val IoU:** 0.5046 (Epoch 46)
- **Best Val Dice:** 0.7027 (Epoch 46)
- **Best Val Accuracy:** 81.29% (Epoch 45)
- **Lowest Val Loss:** 0.3603 (Epoch 48)

---

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory

Reduce batch size in the configuration:

```python
batch_size = 2  # Reduce from 4
```

#### DINOv2 Download Fails

Manually download from:
```bash
pip install torch torchvision --upgrade
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

#### Model File Not Found

Ensure `segmentation_head.pth` exists at:
```
modal/Offroad_Segmentation_Scripts/segmentation_head.pth
```

#### Gradio Interface Not Loading

Check that port 7860 is available:
```bash
netstat -ano | findstr 7860
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


---

## 🙏 Acknowledgments

- **DINOv2** by Meta AI Research for the foundation model
- **Gradio** for the web interface framework
- **PyTorch** for the deep learning framework

---

<p align="center">
  <b>Team SentinalVision</b><br>
  <i>Enabling Safe Autonomous Navigation in Unstructured Environments</i>
</p>
