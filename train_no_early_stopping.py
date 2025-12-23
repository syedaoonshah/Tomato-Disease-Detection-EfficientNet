"""
Training Script WITHOUT Aggressive Early Stopping
This will let models train to full potential
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

sys.path.append('src')

from data_preprocessing import create_data_generators
from model_resnet50 import create_resnet50_model
from model_mobilenetv2 import create_mobilenetv2_model

# Hyperparameters
EPOCHS = 200
BATCH_SIZE = 32
INITIAL_LR = 0.001

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus}")
    except RuntimeError as e:
        print(e)

def train_model_full(model, model_name, train_gen, val_gen, use_early_stopping=False):
    """
    Train model with REDUCED early stopping or NO early stopping.
    """
    
    model.compile(
        optimizer=Adam(learning_rate=INITIAL_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n{'='*60}")
    print(f"Training {model_name} (Full Training)")
    print(f"{'='*60}")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    
    os.makedirs(f'checkpoints/{model_name}_full', exist_ok=True)
    
    # Callbacks
    checkpoint_best = ModelCheckpoint(
        f'models/{model_name}_full.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    steps_per_epoch = len(train_gen)
    checkpoint_periodic = ModelCheckpoint(
        f'checkpoints/{model_name}_full/epoch_{{epoch:03d}}.h5',
        save_freq=10 * steps_per_epoch,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,  # Increased from 10
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint_best, checkpoint_periodic, reduce_lr]
    
    # OPTIONAL: Very relaxed early stopping (or remove entirely)
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=40,  # Much higher! (was 20)
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        print("Early stopping: ENABLED (patience=40)")
    else:
        print("Early stopping: DISABLED (will train full 200 epochs)")
    
    # TensorBoard
    log_dir = f"logs/{model_name}_full/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks.append(tensorboard)
    
    # Train
    print(f"\nStarting full training...")
    print(f"Best model will be saved to: models/{model_name}_full.h5")
    print(f"Checkpoints every 10 epochs to: checkpoints/{model_name}_full/")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n{model_name} training complete!")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return history

def main():
    """Main training function."""
    
    print("\n" + "="*60)
    print("FULL TRAINING (REDUCED/NO EARLY STOPPING)")
    print("="*60)
    print("This training uses:")
    print("  - NO early stopping (trains full 200 epochs)")
    print("  - ReduceLROnPlateau patience=15 (was 10)")
    print("  - Expected: ResNet50 88-90%, MobileNetV2 95-97%")
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Train ResNet50 with NO early stopping
    print("\n" + "="*60)
    print("STEP 1: Training ResNet50 (Full Training)")
    print("="*60)
    
    resnet_model, _ = create_resnet50_model(num_classes=train_gen.num_classes)
    resnet_history = train_model_full(
        resnet_model, 
        'resnet50_model', 
        train_gen, 
        val_gen,
        use_early_stopping=False  # NO early stopping
    )
    
    # Clear memory
    del resnet_model
    tf.keras.backend.clear_session()
    
    # Train MobileNetV2 with NO early stopping
    print("\n" + "="*60)
    print("STEP 2: Training MobileNetV2 (Full Training)")
    print("="*60)
    
    mobilenet_model, _ = create_mobilenetv2_model(num_classes=train_gen.num_classes)
    mobilenet_history = train_model_full(
        mobilenet_model, 
        'mobilenetv2_model', 
        train_gen, 
        val_gen,
        use_early_stopping=False  # NO early stopping
    )
    
    # Summary
    print("\n" + "="*60)
    print("FULL TRAINING COMPLETE!")
    print("="*60)
    print("\nBest Validation Accuracies:")
    print(f"ResNet50:     {max(resnet_history.history['val_accuracy']):.4f}")
    print(f"MobileNetV2:  {max(mobilenet_history.history['val_accuracy']):.4f}")
    
    print("\nSaved models:")
    print("  - models/resnet50_model_full.h5")
    print("  - models/mobilenetv2_model_full.h5")

if __name__ == '__main__':
    main()
