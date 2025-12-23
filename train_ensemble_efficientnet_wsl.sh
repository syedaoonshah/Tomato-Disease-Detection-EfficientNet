#!/bin/bash

# WSL GPU Training Script for Ensemble (MobileNetV2 + EfficientNetB0)

echo "=================================================="
echo "Ensemble Training: MobileNetV2 + EfficientNetB0"
echo "=================================================="

# Set CUDA paths for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Verify GPU
nvidia-smi

# Run training
cd /mnt/d/Tomato
python3 train_ensemble_efficientnet.py
