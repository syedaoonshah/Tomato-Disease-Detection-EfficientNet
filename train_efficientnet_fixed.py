"""
EfficientNetB0 Training - FIXED VERSION
Correct learning rate to prevent overfitting
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from datetime import datetime

sys.path.append('src')
from model_efficientnet import create_efficientnet_model
from data_preprocessing import create_data_generators

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def train_efficientnet_fixed():
    """Train EfficientNetB0 with correct learning rate."""
    
    print("\n" + "="*60)
    print("EFFICIENTNETB0 TRAINING (FIXED)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Create model
    print("\nCreating EfficientNetB0 model...")
    model, base_model = create_efficientnet_model(num_classes=train_gen.num_classes)
    
    # Unfreeze base
    print("\n🔓 Unfreezing base model...")
    base_model.trainable = True
    
    # ✅ USE VERY LOW LEARNING RATE (10x lower than before!)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # ← 0.00001 (was 0.0001)
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/model.count_params()*100:.1f}%)")
    print(f"Learning rate: 1e-5 (VERY LOW to prevent overfitting)")
    
    # Callbacks with MORE AGGRESSIVE regularization
    checkpoint = ModelCheckpoint(
        'models/efficientnet_weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=True
    )
    
    steps_per_epoch = len(train_gen)
    checkpoint_periodic = ModelCheckpoint(
        'checkpoints/efficientnet_model/epoch_{epoch:03d}.h5',
        save_freq=10 * steps_per_epoch,
        verbose=1,
        save_weights_only=True
    )
    
    # More aggressive LR reduction
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # ← Reduced from 15
        min_lr=1e-8,
        verbose=1
    )
    
    log_dir = f"logs/efficientnet_fixed/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    os.makedirs('checkpoints/efficientnet_model', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING TRAINING (FIXED)")
    print("="*60)
    print(f"Learning rate: 1e-5 (prevents overfitting)")
    print(f"Expected: 92-96% validation")
    print(f"Epochs: 200")
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=200,
        callbacks=[checkpoint, checkpoint_periodic, reduce_lr, tensorboard],
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    
    # Save full model
    print("\nSaving full model...")
    model.load_weights('models/efficientnet_weights.h5')
    model.save('models/efficientnet_model.h5', include_optimizer=False, save_format='h5')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    
    if best_val_acc > 0.92:
        print("\n✅ EXCELLENT! Above 92%")
    elif best_val_acc > 0.88:
        print("\n✅ GOOD! Above 88%")
    
    return history

if __name__ == '__main__':
    train_efficientnet_fixed()
