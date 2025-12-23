"""
Evaluate EfficientNetB0 model on test set
Can run while training is ongoing (reads saved weights file)
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

def evaluate_model():
    """Evaluate the trained EfficientNetB0 model."""
    
    print("\n" + "="*60)
    print("EVALUATING EFFICIENTNETB0 ON TEST SET")
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
    print("\nCreating model...")
    model, base_model = create_efficientnet_model(num_classes=10)
    base_model.trainable = True
    
    # Load best weights
    weights_file = 'models/efficientnet_weights.h5'
    if not os.path.exists(weights_file):
        print(f"❌ Weights file not found: {weights_file}")
        print("Training may not have started yet or using different filename.")
        return
    
    print(f"Loading weights from: {weights_file}")
    model.load_weights(weights_file)
    
    # Compile (required for evaluation)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
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
    
    # Detailed predictions
    print("\nGenerating predictions for detailed metrics...")
    y_pred = model.predict(test_gen, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    
    # Classification report
    class_names = list(test_gen.class_indices.keys())
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred_classes, target_names=class_names))
    
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
        print(f"{class_name:30s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    # Save confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'EfficientNetB0 Confusion Matrix\nTest Accuracy: {test_acc*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/efficientnet_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Confusion matrix saved to: results/confusion_matrices/efficientnet_confusion_matrix.png")
    
    return test_acc

if __name__ == '__main__':
    evaluate_model()
