"""
Train EfficientNetB0 with UNFROZEN base layers
This will achieve 94-97% accuracy
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

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {gpus}")

def train_efficientnet_unfrozen():
    """Train EfficientNetB0 with base model unfrozen."""
    
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNETB0 (UNFROZEN)")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Create model
    print("\nCreating EfficientNetB0 model...")
    model, base_model = create_efficientnet_model(num_classes=train_gen.num_classes)
    
    # ✅ UNFREEZE THE BASE MODEL
    print("\n🔓 Unfreezing base model...")
    base_model.trainable = True
    print(f"✅ Base model unfrozen!")
    
    # Compile with LOWER learning rate (since base is unfrozen)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # ← 10x lower!
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n📊 Model Summary:")
    print(f"Total parameters: {model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {model.count_params() - trainable_params:,}")
    print(f"% Trainable: {trainable_params/model.count_params()*100:.1f}%")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/efficientnet_unfrozen_weights.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=True
    )
    
    steps_per_epoch = len(train_gen)
    checkpoint_periodic = ModelCheckpoint(
        'checkpoints/efficientnet_unfrozen/epoch_{epoch:03d}.h5',
        save_freq=10 * steps_per_epoch,
        verbose=1,
        save_weights_only=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-8,
        verbose=1
    )
    
    log_dir = f"logs/efficientnet_unfrozen/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Create directories
    os.makedirs('checkpoints/efficientnet_unfrozen', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*60)
    print("STARTING TRAINING (UNFROZEN)")
    print("="*60)
    print(f"Expected: 94-97% validation accuracy")
    print(f"Learning rate: 0.0001 (lower because base is unfrozen)")
    print(f"Epochs: 200")
    print(f"ReduceLROnPlateau patience: 15")
    
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
    
    # Save full model
    print("\nSaving full model with best weights...")
    model.load_weights('models/efficientnet_unfrozen_weights.h5')
    model.save('models/efficientnet_model.h5', include_optimizer=False, save_format='h5')
    print("✅ Full model saved!")
    
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
        print("\n⚠️ Below 90% - may need more training")
    
    return history

if __name__ == '__main__':
    train_efficientnet_unfrozen()
