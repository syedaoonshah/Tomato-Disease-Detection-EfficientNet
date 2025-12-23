#!/bin/bash
# Resume training from checkpoint with GPU in WSL2

# Set up CUDA library paths
CUDNN_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cudnn/lib
CUDA11_PATH=/home/syed/.local/lib/python3.8/site-packages/nvidia/cublas/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cuda_runtime/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cufft/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/curand/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusolver/lib:/home/syed/.local/lib/python3.8/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=$CUDNN_PATH:$CUDA11_PATH:$LD_LIBRARY_PATH

# Navigate to project
cd /mnt/d/Tomato

# Resume training
echo "=========================================="
echo "Resuming Training from Checkpoint"
echo "=========================================="
python3 -c "
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import sys
import os
import datetime

sys.path.append('src')
from data_preprocessing import create_data_generators

# Enable GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'GPU available: {gpus}')

# Load data
print('Loading data...')
train_gen, val_gen = create_data_generators()

# Find latest checkpoint
import glob
checkpoints = glob.glob('checkpoints/resnet50_model/epoch_*.h5')
if checkpoints:
    # Sort checkpoints and get the latest
    checkpoints.sort()
    latest_checkpoint = checkpoints[-1]
    # Extract epoch number from filename
    epoch_num = int(latest_checkpoint.split('_')[-1].replace('.h5', ''))
    print(f'Found checkpoint at epoch {epoch_num}: {latest_checkpoint}')
else:
    # Fallback to best model
    latest_checkpoint = 'models/resnet50_model.h5'
    epoch_num = 0
    print(f'No periodic checkpoint found, loading best model: {latest_checkpoint}')

# Load existing model
print(f'Loading checkpoint: {latest_checkpoint}')
model = load_model(latest_checkpoint)

# Compile
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Setup callbacks
os.makedirs('checkpoints/resnet50_model', exist_ok=True)

checkpoint_best = ModelCheckpoint(
    'models/resnet50_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

steps_per_epoch = len(train_gen)
checkpoint_periodic = ModelCheckpoint(
    'checkpoints/resnet50_model/epoch_{epoch:03d}.h5',
    save_freq=10 * steps_per_epoch,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

log_dir = 'logs/resnet50_model/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Resume training from latest checkpoint
print(f'\\nResuming training from epoch {epoch_num + 1}...')
history = model.fit(
    train_gen,
    validation_data=val_gen,
    initial_epoch=epoch_num,  # 0-indexed
    epochs=200,
    callbacks=[checkpoint_best, checkpoint_periodic, reduce_lr, early_stop, tensorboard],
    verbose=1
)

print('\\nTraining complete!')
print(f'Best validation accuracy: {max(history.history[\"val_accuracy\"]):.4f}')
"