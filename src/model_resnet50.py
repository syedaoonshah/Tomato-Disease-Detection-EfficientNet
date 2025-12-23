"""
Fine-Tuned ResNet50 Model
Architecture:
1. ResNet50 base (pretrained on ImageNet, exclude top)
2. GlobalAveragePooling2D
3. BatchNormalization
4. Dropout (rate=0.3)
5. Dense (units=2048, for feature extraction)
6. Output: Feature vector (2048-dimensional)

For classification:
7. BatchNormalization
8. Dropout (rate=0.3)
9. Dense(512, activation='relu')
10. Dropout (rate=0.3)
11. Dense(num_classes, activation='softmax')
"""

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, 
    BatchNormalization, 
    Dropout, 
    Dense
)

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Create full ResNet50 model for classification.
    
    Args:
        input_shape: Input image shape (default: 224x224x3)
        num_classes: Number of output classes (default: 10)
    
    Returns:
        model: Complete ResNet50 model
        base_model: Base ResNet50 model (for fine-tuning)
    """
    # Load pretrained ResNet50 (exclude top layers)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Build custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(2048, activation='relu', name='resnet_feature_dense')(x)
    
    # Classification layers
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model

def create_resnet50_feature_extractor(input_shape=(224, 224, 3)):
    """
    Create ResNet50 feature extractor for ensemble model.
    Returns 2048-dimensional feature vector.
    
    Args:
        input_shape: Input image shape (default: 224x224x3)
    
    Returns:
        model: Feature extraction model
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    features = Dense(2048, activation='relu', name='resnet_features')(x)
    
    model = Model(inputs=base_model.input, outputs=features)
    return model

if __name__ == '__main__':
    # Test model creation
    print("Creating ResNet50 model...")
    model, base = create_resnet50_model()
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    
    print("\nCreating ResNet50 feature extractor...")
    feature_model = create_resnet50_feature_extractor()
    print(f"Feature extractor created successfully!")
    print(f"Output shape: {feature_model.output_shape}")
