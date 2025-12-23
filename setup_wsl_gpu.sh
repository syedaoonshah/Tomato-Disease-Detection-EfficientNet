#!/bin/bash
# WSL2 GPU Setup Script for Tomato Disease Classification

echo "=========================================="
echo "WSL2 GPU Setup for TensorFlow"
echo "=========================================="

# Update system
echo "Step 1/5: Updating system..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
echo "Step 2/5: Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3-pip

# Create virtual environment
echo "Step 3/5: Creating virtual environment..."
cd /mnt/d/Tomato
python3.11 -m venv .venv-wsl

# Activate and install packages
echo "Step 4/5: Installing TensorFlow with GPU support..."
source .venv-wsl/bin/activate
pip install --upgrade pip
pip install tensorflow[and-cuda]
pip install scikit-learn matplotlib seaborn pillow

# Verify GPU
echo "Step 5/5: Verifying GPU detection..."
python -c "import tensorflow as tf; print('\\n=== GPU Detection ==='); print('TensorFlow version:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', gpus); print('GPU available:', len(gpus) > 0); print('\\nSetup complete! GPU is', 'READY' if len(gpus) > 0 else 'NOT DETECTED')"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "1. Open WSL: wsl"
echo "2. Navigate: cd /mnt/d/Tomato"
echo "3. Activate: source .venv-wsl/bin/activate"
echo "4. Train: python src/train.py"
echo ""
