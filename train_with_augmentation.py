"""
Train ResNet50 with Data Augmentation
Alternative approach to boost accuracy using augmented training data
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from src.model_resnet50 import create_resnet50_model
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_PATH = 'dataset_proper_split/train'
VAL_PATH = 'dataset_proper_split/val'

print("Setting up GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {gpus}")
    except RuntimeError as e:
        print(e)

def main():
    # AUGMENTED training data generator
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,           # Random rotation ±20 degrees
        width_shift_range=0.2,       # Horizontal shift
        height_shift_range=0.2,      # Vertical shift
        shear_range=0.2,            # Shear transformation
        zoom_range=0.2,             # Random zoom
        horizontal_flip=True,        # Random horizontal flip
        fill_mode='nearest'
    )
    
    # Validation without augmentation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_gen = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    val_gen = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    print(f"\n{'='*60}")
    print("TRAINING ResNet50 WITH DATA AUGMENTATION")
    print(f"{'='*60}")
    print(f"Training samples: {train_gen.samples:,}")
    print(f"Validation samples: {val_gen.samples:,}")
    print(f"Augmentation: rotation, shift, shear, zoom, flip")
    
    # Create model
    model, base_model = create_resnet50_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    checkpoint_best = ModelCheckpoint(
        'models/resnet50_augmented.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-7,
        verbose=1
    )
    
    log_dir = f"logs/resnet50_augmented/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard = TensorBoard(log_dir=log_dir)
    
    callbacks = [checkpoint_best, reduce_lr, tensorboard]
    
    # Train
    print("\nStarting training with augmentation...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Model saved to: models/resnet50_augmented.h5")

if __name__ == '__main__':
    main()
