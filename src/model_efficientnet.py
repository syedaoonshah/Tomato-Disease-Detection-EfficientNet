"""
EfficientNetB0 Model for Tomato Leaf Disease Classification
"""

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout,
    Dense
)

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create EfficientNetB0 model.
    
    Args:
        input_shape: Input image shape (default: 224x224x3)
        num_classes: Number of output classes (default: 10)
    
    Returns:
        model: Complete EfficientNetB0 model
        base_model: Base model (for fine-tuning)
    """
    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base initially
    base_model.trainable = False
    
    # Build custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1280, activation='relu', name='efficientnet_feature_dense')(x)
    
    # Classification layers
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model

def create_efficientnet_feature_extractor(input_shape=(224, 224, 3)):
    """
    Create EfficientNetB0 feature extractor for ensemble.
    Returns 1280-dimensional feature vector.
    """
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    features = Dense(1280, activation='relu', name='efficientnet_features')(x)
    
    model = Model(inputs=base_model.input, outputs=features)
    return model

if __name__ == '__main__':
    import tensorflow as tf
    
    print("Creating EfficientNetB0 model...")
    model, base = create_efficientnet_model()
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
