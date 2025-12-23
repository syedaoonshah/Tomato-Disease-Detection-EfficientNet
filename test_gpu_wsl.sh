#!/bin/bash
CUDNN_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cudnn/lib
CUDA11_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cublas/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cufft/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/curand/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA11_PATH:$LD_LIBRARY_PATH
cd /mnt/d/Tomato
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__); gpus = tf.config.list_physical_devices('GPU'); print('GPU devices:', gpus); print('GPU available:', len(gpus) > 0)"
