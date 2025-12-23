#!/bin/bash

echo "=========================================="
echo "EfficientNetB0 Training (FIXED LR) with GPU"
echo "=========================================="

# Set CUDA environment
export LD_LIBRARY_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cudnn/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cublas/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cufft/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/curand/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH

cd /mnt/d/Tomato

python3 train_efficientnet_fixed.py
