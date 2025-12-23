"""
Train Ensemble Model: MobileNetV2 + EfficientNetB0
Creates a meta-model that learns to combine predictions
CORRECTED VERSION
"""

import os
import sys
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
import numpy as np

sys.path.append('src')
from data_preprocessing import create_data_generators
from model_efficientnet import create_efficientnet_model

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU available: {gpus[0].name}")
    except RuntimeError as e:
        print(e)

def create_ensemble_model(num_classes=10):
    """
    Create ensemble model by concatenating features from both models.
    """
    
    print("\n" + "="*60)
    print("LOADING BASE MODELS")
    print("="*60)
    
    # Load MobileNetV2
    print("\nLoading MobileNetV2...")
    mobilenet_full = load_model('models/mobilenetv2_model.h5')
    
    # Find the feature layer (before final Dense layer)
    # Look for the last Dense layer with relu activation
    mobilenet_feature_layer = None
    for i in range(len(mobilenet_full.layers) - 1, -1, -1):
        layer = mobilenet_full.layers[i]
        if isinstance(layer, layers.Dense) and layer.activation.__name__ == 'relu':
            mobilenet_feature_layer = layer.output
            print(f"   Found feature layer: {layer.name} (shape: {layer.output.shape})")
            break
    
    if mobilenet_feature_layer is None:
        # Fallback: use layer before final softmax
        mobilenet_feature_layer = mobilenet_full.layers[-2].output
        print(f"   Using fallback: {mobilenet_full.layers[-2].name}")
    
    mobilenet_base = Model(
        inputs=mobilenet_full.input,
        outputs=mobilenet_feature_layer,
        name='mobilenet_base'
    )
    mobilenet_base.trainable = False
    print(f"✅ MobileNetV2 feature extractor created")
    
    # Load EfficientNetB0
    print("\nLoading EfficientNetB0...")
    efficientnet_full, _ = create_efficientnet_model(num_classes=10)
    efficientnet_full.load_weights('models/efficientnet_weights.h5')
    efficientnet_full.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Find the feature layer for EfficientNetB0
    efficientnet_feature_layer = None
    for i in range(len(efficientnet_full.layers) - 1, -1, -1):
        layer = efficientnet_full.layers[i]
        if isinstance(layer, layers.Dense) and layer.activation.__name__ == 'relu':
            efficientnet_feature_layer = layer.output
            print(f"   Found feature layer: {layer.name} (shape: {layer.output.shape})")
            break
    
    if efficientnet_feature_layer is None:
        efficientnet_feature_layer = efficientnet_full.layers[-2].output
        print(f"   Using fallback: {efficientnet_full.layers[-2].name}")
    
    efficientnet_base = Model(
        inputs=efficientnet_full.input,
        outputs=efficientnet_feature_layer,
        name='efficientnet_base'
    )
    efficientnet_base.trainable = False
    print(f"✅ EfficientNetB0 feature extractor created")
    
    print("\n" + "="*60)
    print("CREATING ENSEMBLE ARCHITECTURE")
    print("="*60)
    
    # Input layer
    input_layer = layers.Input(shape=(224, 224, 3), name='ensemble_input')
    
    # Get features from both models
    mobilenet_features = mobilenet_base(input_layer)
    efficientnet_features = efficientnet_base(input_layer)
    
    print(f"\nFeature shapes:")
    print(f"  MobileNetV2:     {mobilenet_features.shape}")
    print(f"  EfficientNetB0:  {efficientnet_features.shape}")
    
    # Concatenate features
    concatenated = layers.Concatenate(name='concat_features')([mobilenet_features, efficientnet_features])
    print(f"  Concatenated:    {concatenated.shape}")
    
    # Meta-learner layers
    x = layers.Dense(256, activation='relu', name='meta_dense1')(concatenated)
    x = layers.BatchNormalization(name='meta_bn1')(x)
    x = layers.Dropout(0.4, name='meta_dropout1')(x)
    
    x = layers.Dense(128, activation='relu', name='meta_dense2')(x)
    x = layers.BatchNormalization(name='meta_bn2')(x)
    x = layers.Dropout(0.3, name='meta_dropout2')(x)
    
    # Final output
    output = layers.Dense(num_classes, activation='softmax', name='ensemble_output')(x)
    
    # Create model
    ensemble_model = Model(inputs=input_layer, outputs=output, name='ensemble_model')
    
    print("\n✅ Ensemble model created")
    print(f"\nArchitecture:")
    print(f"  Input → MobileNetV2 (frozen)")
    print(f"       → EfficientNetB0 (frozen)")
    print(f"       → Concatenate")
    print(f"       → Dense(256) + BN + Dropout(0.4)")
    print(f"       → Dense(128) + BN + Dropout(0.3)")
    print(f"       → Dense(10, softmax)")
    
    return ensemble_model

def train_ensemble():
    """Train the ensemble model"""
    
    print("\n" + "="*60)
    print("ENSEMBLE MODEL TRAINING")
    print("MobileNetV2 (95%) + EfficientNetB0 (99.4%)")
    print("="*60)
    
    # Create ensemble
    model = create_ensemble_model(num_classes=10)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Model summary
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    frozen_params = total_params - trainable_params
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_gen, val_gen = create_data_generators()
    
    print(f"Training samples: {train_gen.n}")
    print(f"Validation samples: {val_gen.n}")
    print(f"Batch size: {train_gen.batch_size}")
    
    # Callbacks
    checkpoint_dir = 'checkpoints/ensemble_efficientnet'
    log_dir = 'logs/ensemble_efficientnet'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath='models/ensemble_efficientnet_best_weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        CSVLogger(
            os.path.join(log_dir, 'training.log'),
            append=False
        )
    ]
    
    # Training configuration
    epochs = 50
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Epochs: {epochs}")
    print(f"Learning rate: 1e-4")
    print(f"Optimizer: Adam")
    print(f"Loss: categorical_crossentropy")
    print(f"ReduceLROnPlateau: patience=5, factor=0.5")
    print(f"EarlyStopping: patience=10")
    print(f"Best weights: models/ensemble_efficientnet_best_weights.h5")
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("\nExpected outcome:")
    print("  - Meta-learner learns optimal feature combination")
    print("  - Target: Match or slightly exceed EfficientNetB0 (98-99%)")
    print("  - Note: With 99.4% EfficientNetB0, ensemble may not improve much")
    print("\n")
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final weights
    model.save_weights('models/ensemble_efficientnet_final_weights.h5')
    print(f"\n✅ Final weights saved: models/ensemble_efficientnet_final_weights.h5")
    
    # Print best validation accuracy
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_val_acc = history.history['val_accuracy'][-1]
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) at epoch {best_epoch}")
    print(f"Final validation accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"\nComparison:")
    print(f"  MobileNetV2:     95.0%")
    print(f"  EfficientNetB0:  99.4%")
    print(f"  Ensemble:        {best_val_acc*100:.2f}%")
    
    if best_val_acc > 0.994:
        print(f"\n✅ EXCELLENT! Ensemble improved over EfficientNetB0!")
    elif best_val_acc >= 0.990:
        print(f"\n✅ GREAT! Ensemble matches EfficientNetB0 performance!")
    else:
        print(f"\n✅ Good result! Ensemble validates individual model performance.")
    
    print(f"\nBest weights: models/ensemble_efficientnet_best_weights.h5")
    
    return history

if __name__ == '__main__':
    train_ensemble()
