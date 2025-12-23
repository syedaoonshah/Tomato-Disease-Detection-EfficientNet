#!/bin/bash
# Train EfficientNetB0 with GPU in WSL2

# Set up CUDA library paths
CUDNN_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cudnn/lib
CUDA11_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cublas/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cufft/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/curand/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA11_PATH:$LD_LIBRARY_PATH

# Navigate to project
cd /mnt/d/Tomato

# Run training
echo "=========================================="
echo "Starting EfficientNetB0 Training with GPU"
echo "=========================================="
python3 train_efficientnet.py
