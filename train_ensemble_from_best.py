"""
Train Ensemble Using BEST Trained Models as Feature Extractors
This uses your trained models (not fresh extractors)
"""

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    Dense,
    BatchNormalization,
    Dropout
)
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

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def create_ensemble_from_best_models(
    resnet_path='models/resnet50_model_full.h5',
    mobilenet_path='models/mobilenetv2_model_full.h5',
    num_classes=10
):
    """
    Create ensemble using YOUR BEST TRAINED models.
    Extracts learned features from your trained models.
    """
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE FROM BEST TRAINED MODELS")
    print("="*60)
    
    # Load your BEST trained models
    print(f"\nLoading trained models...")
    resnet_trained = load_model(resnet_path)
    mobilenet_trained = load_model(mobilenet_path)
    
    print(f"✅ Loaded ResNet50 from {resnet_path}")
    print(f"✅ Loaded MobileNetV2 from {mobilenet_path}")
    
    # Print model structures to find feature layers
    print(f"\nResNet50 layers: {len(resnet_trained.layers)}")
    print(f"MobileNetV2 layers: {len(mobilenet_trained.layers)}")
    
    # Find the feature extraction layer (before final classification)
    # For ResNet50: layers are [..., Dense(512), Dropout, Dense(10, softmax)]
    # We want the Dense(512) output
    resnet_feature_layer_idx = -4  # 4 layers before end
    mobilenet_feature_layer_idx = -4  # 4 layers before end
    
    print(f"\nExtracting features from layer {resnet_feature_layer_idx} of ResNet50")
    print(f"Extracting features from layer {mobilenet_feature_layer_idx} of MobileNetV2")
    
    # Create input
    input_layer = Input(shape=(224, 224, 3))
    
    # Create feature extractors from trained models
    resnet_extractor = Model(
        inputs=resnet_trained.input,
        outputs=resnet_trained.layers[resnet_feature_layer_idx].output
    )
    
    mobilenet_extractor = Model(
        inputs=mobilenet_trained.input,
        outputs=mobilenet_trained.layers[mobilenet_feature_layer_idx].output
    )
    
    # Freeze the extractors (use learned features)
    resnet_extractor.trainable = False
    mobilenet_extractor.trainable = False
    
    print(f"\n✅ Created feature extractors")
    print(f"   ResNet50 output shape: {resnet_extractor.output_shape}")
    print(f"   MobileNetV2 output shape: {mobilenet_extractor.output_shape}")
    
    # Extract features
    resnet_features = resnet_extractor(input_layer)
    mobilenet_features = mobilenet_extractor(input_layer)
    
    # Concatenate
    concatenated = Concatenate()([resnet_features, mobilenet_features])
    
    print(f"   Concatenated shape: {concatenated.shape}")
    
    # New classification layers
    x = Dense(1024, activation='relu')(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs)
    
    return model

def main():
    """Train ensemble using best models."""
    
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING FROM BEST MODELS")
    print("="*60)
    
    # Check if best models exist
    resnet_path = 'models/resnet50_model_full.h5'
    mobilenet_path = 'models/mobilenetv2_model_full.h5'
    
    if not os.path.exists(resnet_path):
        resnet_path = 'models/resnet50_model.h5'  # Fallback to original
        print(f"⚠️  Using original ResNet50 model (not _full.h5)")
    if not os.path.exists(mobilenet_path):
        mobilenet_path = 'models/mobilenetv2_model.h5'  # Fallback to original
        print(f"⚠️  Using original MobileNetV2 model (not _full.h5)")
    
    print(f"\nUsing models:")
    print(f"  ResNet50: {resnet_path}")
    print(f"  MobileNetV2: {mobilenet_path}")
    
    # Load data
    print("\nLoading data...")
    train_gen, val_gen = create_data_generators()
    
    # Create ensemble from best models
    ensemble_model = create_ensemble_from_best_models(
        resnet_path=resnet_path,
        mobilenet_path=mobilenet_path,
        num_classes=train_gen.num_classes
    )
    
    # Compile
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\nEnsemble model:")
    print(f"  Total parameters: {ensemble_model.count_params():,}")
    trainable_params = sum([tf.keras.backend.count_params(w) for w in ensemble_model.trainable_weights])
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/ensemble_model_best.h5',
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
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,  # Relaxed
        restore_best_weights=True,
        verbose=1
    )
    
    log_dir = f"logs/ensemble_best/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING ENSEMBLE")
    print("="*60)
    print("This trains only the classification layers on top")
    print("Feature extractors are frozen (using learned features)")
    
    history = ensemble_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100,  # Less epochs since we're only training top layers
        callbacks=[checkpoint, reduce_lr, early_stop, tensorboard],
        verbose=1
    )
    
    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Saved to: models/ensemble_model_best.h5")

if __name__ == '__main__':
    main()
