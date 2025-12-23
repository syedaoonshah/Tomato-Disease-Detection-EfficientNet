"""
Training Script for Tomato Leaf Disease Classification

Hyperparameters (from paper):
- Optimizer: Adam
- Batch size: 32
- Epochs: 200
- Learning rate: Use ReduceLROnPlateau scheduler
- Loss: Categorical Crossentropy
- Metrics: Accuracy

Training Strategy:
1. Train each model individually
2. Save best weights
3. Use saved models for ensemble
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard
)
import sys
import os
import datetime

sys.path.append(os.path.dirname(__file__))

from data_preprocessing import create_data_generators
from model_resnet50 import create_resnet50_model
from model_mobilenetv2 import create_mobilenetv2_model
from model_ensemble import create_ensemble_model

# Hyperparameters
EPOCHS = 200
BATCH_SIZE = 32
INITIAL_LR = 0.001

# GPU Configuration for RTX 4060
print("Setting up GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"CUDA available: {tf.test.is_built_with_cuda()}")
        print(f"GPU device: {tf.test.gpu_device_name()}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Training will use CPU (slower).")

def train_model(model, model_name, train_gen, val_gen):
    """
    Train a single model.
    
    Args:
        model: Keras model to train
        model_name: Name for saving checkpoints
        train_gen: Training data generator
        val_gen: Validation data generator
    
    Returns:
        history: Training history
    """
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {model.count_params() - trainable_params:,}")
    
    # Create checkpoint directory
    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)
    
    # Callbacks
    # Save best model
    checkpoint_best = ModelCheckpoint(
        f'models/{model_name}.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Save checkpoint every 10 epochs
    steps_per_epoch = len(train_gen)
    checkpoint_periodic = ModelCheckpoint(
        f'checkpoints/{model_name}/epoch_{{epoch:03d}}.h5',
        save_freq=10 * steps_per_epoch,  # Save every 10 epochs
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # Optimized for faster convergence
        min_lr=1e-7,
        verbose=1
    )
    
    # NO EARLY STOPPING - Train full 200 epochs
    # early_stop = EarlyStopping(
    #     monitor='val_loss',
    #     patience=20,
    #     restore_best_weights=True,
    #     verbose=1
    # )
    
    # TensorBoard for monitoring
    log_dir = f"logs/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train
    print(f"\nStarting training...")
    print(f"ReduceLROnPlateau patience: 15 epochs")
    print(f"Early stopping: DISABLED (will train full 200 epochs)")
    print(f"Checkpoints will be saved every 10 epochs to: checkpoints/{model_name}/")
    print(f"Best model will be saved to: models/{model_name}.h5")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[checkpoint_best, checkpoint_periodic, reduce_lr, tensorboard],  # Removed early_stop
        verbose=1
    )
    
    print(f"\n{model_name} training complete!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return history

def main():
    """Main training function."""
    
    print("\n" + "="*60)
    print("TOMATO LEAF DISEASE CLASSIFICATION - TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    print(f"\nDataset Information:")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: 224x224")
    print(f"\nClass distribution:")
    for class_name, idx in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {class_name}")
    
    # Train ResNet50
    print("\n" + "="*60)
    print("STEP 1/3: Training ResNet50")
    print("="*60)
    resnet_model, _ = create_resnet50_model(num_classes=train_gen.num_classes)
    resnet_history = train_model(resnet_model, 'resnet50_model', train_gen, val_gen)
    
    # Clear memory
    del resnet_model
    tf.keras.backend.clear_session()
    
    # Train MobileNetV2
    print("\n" + "="*60)
    print("STEP 2/3: Training MobileNetV2")
    print("="*60)
    mobilenet_model, _ = create_mobilenetv2_model(num_classes=train_gen.num_classes)
    mobilenet_history = train_model(mobilenet_model, 'mobilenetv2_model', train_gen, val_gen)
    
    # Clear memory
    del mobilenet_model
    tf.keras.backend.clear_session()
    
    # Train Ensemble
    print("\n" + "="*60)
    print("STEP 3/3: Training Ensemble Model")
    print("="*60)
    ensemble_model = create_ensemble_model(num_classes=train_gen.num_classes)
    ensemble_history = train_model(ensemble_model, 'ensemble_model', train_gen, val_gen)
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\nBest Validation Accuracies:")
    print(f"ResNet50:     {max(resnet_history.history['val_accuracy']):.4f}")
    print(f"MobileNetV2:  {max(mobilenet_history.history['val_accuracy']):.4f}")
    print(f"Ensemble:     {max(ensemble_history.history['val_accuracy']):.4f}")
    
    print("\nSaved models:")
    print("  - models/resnet50_model.h5")
    print("  - models/mobilenetv2_model.h5")
    print("  - models/ensemble_model.h5")
    
    print("\nNext step: Run 'python src/evaluate.py' to evaluate the models")

if __name__ == '__main__':
    main()
