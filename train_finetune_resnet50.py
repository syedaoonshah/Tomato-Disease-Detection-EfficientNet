"""
Fine-Tune ResNet50 - Unfreeze and train with low learning rate
This is the MISSING STEP to reach 88-90%
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard
)
import datetime
import sys
import os
sys.path.append('src')
from data_preprocessing import create_data_generators

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def finetune_resnet50():
    """
    Fine-tune ResNet50 by unfreezing base layers.
    """
    
    print("\n" + "="*60)
    print("FINE-TUNING RESNET50")
    print("="*60)
    print("Strategy: Unfreeze base layers + train with low LR")
    
    # Load your trained model (80% validation)
    print("\nLoading trained ResNet50 model...")
    model = load_model('models/resnet50_model.h5')
    print(f"✅ Model loaded (current validation: ~80%)")
    
    # Find the base ResNet50 layers
    print("\nModel structure:")
    print(f"Total layers: {len(model.layers)}")
    
    # Find ResNet50 base model
    base_model = None
    for layer in model.layers:
        if 'resnet50' in layer.name.lower() or isinstance(layer, tf.keras.Model):
            if hasattr(layer, 'layers') and len(layer.layers) > 50:
                base_model = layer
                break
    
    if base_model is None:
        print("⚠️ Could not find ResNet50 base model automatically")
        print("Attempting to unfreeze last 50 layers of entire model...")
        print(f"\nCurrent trainable status:")
        trainable_count = sum([1 for l in model.layers if l.trainable])
        print(f"Trainable layers: {trainable_count}/{len(model.layers)}")
        
        # Unfreeze last 50 layers
        for layer in model.layers[-50:]:
            layer.trainable = True
        
        print(f"\nAfter unfreezing last 50 layers:")
        trainable_count = sum([1 for l in model.layers if l.trainable])
        print(f"Trainable layers: {trainable_count}/{len(model.layers)}")
    else:
        print(f"\n✅ Found ResNet50 base model: {base_model.name}")
        print(f"   Base layers: {len(base_model.layers)}")
        
        # Unfreeze last 50 layers of ResNet50 base
        print(f"\nUnfreezing last 50 layers of ResNet50 base...")
        for layer in base_model.layers[-50:]:
            layer.trainable = True
        
        trainable_base = sum([1 for l in base_model.layers if l.trainable])
        print(f"✅ Unfrozen: {trainable_base}/{len(base_model.layers)} base layers")
    
    # Count total trainable parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    total_params = model.count_params()
    
    print(f"\n📊 Parameter Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Recompile with VERY LOW learning rate
    print("\nRecompiling model with low learning rate...")
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Very low! (was 1e-3)
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print(f"Learning rate: 1e-5 (100x lower than initial 1e-3)")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/resnet50_finetuned.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-8,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    log_dir = f"logs/resnet50_finetuned/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Create checkpoint directory
    os.makedirs('checkpoints/resnet50_finetune', exist_ok=True)
    
    # Fine-tune
    print("\n" + "="*60)
    print("STARTING FINE-TUNING")
    print("="*60)
    print("Expected: 80% → 88-90% validation accuracy")
    print("Training with unfrozen base layers + low LR")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[checkpoint, reduce_lr, early_stop, tensorboard],
        verbose=1
    )
    
    # Results
    best_val_acc = max(history.history['val_accuracy'])
    
    print("\n" + "="*60)
    print("FINE-TUNING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"Saved to: models/resnet50_finetuned.h5")
    
    if best_val_acc > 0.87:
        print("\n✅ SUCCESS! Achieved target >87%")
    elif best_val_acc > 0.85:
        print("\n✅ Good improvement! Above 85%")
    else:
        print("\n⚠️ Below target. Consider:")
        print("  - Training for more epochs")
        print("  - Unfreezing more layers")
        print("  - Using data augmentation")
    
    return history

if __name__ == '__main__':
    finetune_resnet50()
