"""
Test Ensemble Model During Training
Reads saved weights without stopping the training process
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')
from model_efficientnet import create_efficientnet_model

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_ensemble_model(num_classes=10):
    """Recreate the ensemble model architecture (must match training)"""
    
    # Load MobileNetV2
    mobilenet_full = load_model('models/mobilenetv2_model.h5')
    mobilenet_base = Model(
        inputs=mobilenet_full.input,
        outputs=mobilenet_full.layers[-2].output,
        name='mobilenet_base'
    )
    mobilenet_base.trainable = False
    
    # Load EfficientNetB0
    efficientnet_full, _ = create_efficientnet_model(num_classes=10)
    efficientnet_full.load_weights('models/efficientnet_weights.h5')
    efficientnet_base = Model(
        inputs=efficientnet_full.input,
        outputs=efficientnet_full.layers[-2].output,
        name='efficientnet_base'
    )
    efficientnet_base.trainable = False
    
    # Input layer
    input_layer = layers.Input(shape=(224, 224, 3), name='ensemble_input')
    
    # Get features from both models
    mobilenet_features = mobilenet_base(input_layer)
    efficientnet_features = efficientnet_base(input_layer)
    
    # Concatenate features
    concatenated = layers.Concatenate(name='concat_features')([mobilenet_features, efficientnet_features])
    
    # Meta-learner layers (must match training exactly)
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
    
    return ensemble_model

def evaluate_model():
    """Evaluate the ensemble model on test set."""
    
    print("\n" + "="*60)
    print("TESTING ENSEMBLE MODEL")
    print("MobileNetV2 + EfficientNetB0")
    print("="*60)
    
    # Load test data
    print("\nLoading test data...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        'dataset_proper_split/test',
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Test samples: {test_gen.n}")
    print(f"Classes: {list(test_gen.class_indices.keys())}")
    
    # Create model
    print("\nRecreating ensemble model architecture...")
    model = create_ensemble_model(num_classes=10)
    
    # Load best weights
    weights_file = 'models/ensemble_efficientnet_best_weights.h5'
    if not os.path.exists(weights_file):
        print(f"❌ Weights file not found: {weights_file}")
        print("Training may not have started yet or using different filename.")
        return
    
    print(f"Loading weights from: {weights_file}")
    model.load_weights(weights_file)
    
    # Compile (required for evaluation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING ON TEST SET")
    print("="*60)
    
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Comparison with individual models
    print("\n" + "="*60)
    print("COMPARISON WITH INDIVIDUAL MODELS")
    print("="*60)
    print(f"MobileNetV2:     92.45%")
    print(f"EfficientNetB0:  98.73%")
    print(f"Ensemble:        {test_acc*100:.2f}%")
    
    if test_acc > 0.9873:
        improvement = (test_acc - 0.9873) * 100
        print(f"\n✅ Ensemble improved over best model by {improvement:.2f}%!")
    else:
        difference = (0.9873 - test_acc) * 100
        print(f"\n⚠️  Ensemble is {difference:.2f}% below EfficientNetB0")
    
    # Detailed predictions
    print("\nGenerating predictions for detailed metrics...")
    test_gen.reset()
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    class_names = list(test_gen.class_indices.keys())
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred_classes, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    print("\n" + "="*60)
    print("PER-CLASS ACCURACY")
    print("="*60)
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        total = cm[i].sum()
        correct = cm[i, i]
        errors = total - correct
        print(f"{class_name:30s}: {class_acc:.4f} ({class_acc*100:.2f}%) - {correct}/{total} correct, {errors} errors")
    
    # Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Ensemble Model Confusion Matrix\nTest Accuracy: {test_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/ensemble_test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: results/confusion_matrices/ensemble_test_confusion_matrix.png")
    
    plt.close()
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    
    return test_acc

if __name__ == '__main__':
    evaluate_model()
