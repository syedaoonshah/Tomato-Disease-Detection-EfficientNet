"""
Data Preprocessing Module - CORRECTED FOR PROPER 80-10-10 SPLIT
- Load images from 10 class folders
- Resize to 224×224 pixels
- Normalize pixel values to [0,1]
- Split: 80% train (8,800), 10% validation (1,100), 10% test (1,100)
- NO data augmentation (rotation, flipping, etc.) as per paper
"""

import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduced from 32 to prevent GPU OOM

# Use the properly split dataset (created by create_proper_split.py)
TRAIN_PATH = 'dataset_proper_split/train'   # 8,800 images
VAL_PATH = 'dataset_proper_split/val'       # 1,100 images
TEST_PATH = 'dataset_proper_split/test'     # 1,100 images

# Image data generators (normalization only - NO augmentation as per paper)
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

def create_data_generators():
    """
    Create data generators using proper 80-10-10 split.
    Training: 8,800 images (80% of 11,000)
    Validation: 1,100 images (10% of 11,000)
    """
    
    if not os.path.exists(TRAIN_PATH):
        raise ValueError(
            f"Proper split not found at {TRAIN_PATH}\n"
            f"Please run: python create_proper_split.py first!"
        )
    
    # Training generator (8,800 images)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    
    # Validation generator (1,100 images)
    validation_generator = val_datagen.flow_from_directory(
        VAL_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    # Verify counts
    print(f"\n{'='*60}")
    print("DATASET VERIFICATION")
    print(f"{'='*60}")
    print(f"Training samples:   {train_generator.samples:,} (expected: 8,800)")
    print(f"Validation samples: {validation_generator.samples:,} (expected: 1,100)")
    print(f"Number of classes:  {train_generator.num_classes} (expected: 10)")
    
    if train_generator.samples != 8800:
        print(f"⚠️  WARNING: Training samples don't match paper!")
    else:
        print(f"✅ Training samples match paper exactly!")
    
    return train_generator, validation_generator

def create_test_generator():
    """
    Create test generator (1,100 images).
    """
    
    if not os.path.exists(TEST_PATH):
        raise ValueError(
            f"Test split not found at {TEST_PATH}\n"
            f"Please run: python create_proper_split.py first!"
        )
    
    test_generator = test_datagen.flow_from_directory(
        TEST_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    
    print(f"\nTest samples: {test_generator.samples:,} (expected: 1,100)")
    
    return test_generator

def get_dataset_info():
    """Get basic dataset information."""
    return {
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'train_path': TRAIN_PATH,
        'val_path': VAL_PATH,
        'test_path': TEST_PATH
    }
