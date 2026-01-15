# 🖊️ SignVerifAI

**Offline Signature Verification using Siamese Networks**

TUBITAK 2209-A Project - AI-powered signature verification system using deep learning.

## 🎯 Features

- **MobileNetV3-Large** backbone with 128-dim embeddings
- **Contrastive Loss** for similarity learning
- **Multi-device support**: CUDA (Colab), MPS (Apple Silicon), CPU
- **CLI interface** for all pipeline stages
- **Comprehensive metrics**: AUC-ROC, EER, FAR/FRR

## 📁 Project Structure

```
SignVerifAI/
├── src/signverify/      # Main package
│   ├── config.py        # Configuration
│   ├── cli.py           # Command-line interface
│   ├── data/            # Data processing modules
│   ├── models/          # Neural network modules
│   └── train/           # Training & evaluation
├── notebooks/           # Colab notebooks
├── data_processed/      # Preprocessed images (not in repo)
├── splits/              # Train/val/test splits (generated)
├── pairs/               # Siamese pairs (generated)
└── outputs/             # Models, logs, reports
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended for Training)

1. Open the notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gorkemelih/SignVerifAI/blob/main/notebooks/SignVerifAI_Colab_Training.ipynb)

2. Set runtime to GPU: `Runtime` → `Change runtime type` → `T4 GPU`

3. Follow notebook instructions

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/gorkemelih/SignVerifAI.git
cd SignVerifAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Run pipeline
signverify info
signverify split
signverify pairs
signverify train --device mps --epochs 50
signverify eval
```

## 📊 CLI Commands

| Command | Description |
|---------|-------------|
| `signverify info` | Show project status |
| `signverify audit` | Audit dataset quality |
| `signverify preprocess` | Preprocess images |
| `signverify split` | Create train/val/test splits |
| `signverify pairs` | Generate Siamese pairs |
| `signverify train` | Train the model |
| `signverify eval` | Evaluate on test set |
| `signverify pipeline` | Run full pipeline |

## 🔧 Training Parameters

```bash
signverify train \
  --device cuda \        # cuda, mps, cpu
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4
```

## 📈 Model Architecture

| Component | Specification |
|-----------|--------------|
| Backbone | MobileNetV3-Large (ImageNet pretrained) |
| Embedding | 128-dimensional, L2 normalized |
| Similarity | Cosine similarity |
| Loss | Contrastive Loss (margin=1.0) |
| Input | 224×224 grayscale → 3-channel RGB |

## 📂 Data Setup

The training data is not included in this repository due to size.

### For Google Colab:
1. Upload `data_processed.zip` to Google Drive
2. Path: `MyDrive/SignVerifAI/data_processed.zip`
3. Run the Colab notebook

### For Local:
1. Place preprocessed images in `data_processed/`
2. Ensure `metadata.csv` exists

## 📄 License

MIT License - TUBITAK 2209-A Project

## 👥 Authors

- Görkem Melih Özcan
