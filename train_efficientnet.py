"""
Train EfficientNetB0 Model
Expected: 94-97% validation accuracy
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from datetime import datetime

sys.path.append('src')
from model_efficientnet import create_efficientnet_model
from data_preprocessing import create_data_generators

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {gpus}")

def train_efficientnet():
    """Train EfficientNetB0 model."""
    
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNETB0")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Create model
    print("\nCreating EfficientNetB0 model...")
    model, base_model = create_efficientnet_model(num_classes=train_gen.num_classes)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {model.count_params() - trainable_params:,}")
    
    # Callbacks - use weights-only during training to avoid JSON serialization error
    checkpoint = ModelCheckpoint(
        'models/efficientnet_model_weights.h5',
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
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    log_dir = f"logs/efficientnet_model/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Create directories
    os.makedirs('checkpoints/efficientnet_model', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Expected: 94-97% validation accuracy")
    print(f"Epochs: 200 (no early stopping)")
    print(f"ReduceLROnPlateau patience: 15")
    print(f"Best model: models/efficientnet_model.h5")
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=200,
        callbacks=[checkpoint, checkpoint_periodic, reduce_lr, tensorboard],
        verbose=1
    )
    
    # Results
    best_val_acc = max(history.history['val_accuracy'])
    
    # Reconstruct model with best weights and save as full model
    print("\nSaving full model with best weights...")
    model.load_weights('models/efficientnet_model_weights.h5')
    model.save('models/efficientnet_model.h5', include_optimizer=False, save_format='h5')
    print("✅ Full model saved successfully!")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Saved to: models/efficientnet_model.h5")
    
    if best_val_acc > 0.94:
        print("\n✅ EXCELLENT! Above 94%")
    elif best_val_acc > 0.90:
        print("\n✅ GOOD! Above 90%")
    else:
        print("\n⚠️ Below 90% - may need fine-tuning")
    
    return history

if __name__ == '__main__':
    train_efficientnet()
