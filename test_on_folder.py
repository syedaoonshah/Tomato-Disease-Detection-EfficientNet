"""
Test EfficientNetB0 model on specific image folder
Usage: python test_on_folder.py <folder_path>
Example: python test_on_folder.py dataset_proper_split/test
         python test_on_folder.py my_custom_images
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

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

def test_on_folder(folder_path):
    """Test the model on a specific folder."""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    print("\n" + "="*60)
    print(f"TESTING EFFICIENTNETB0 ON: {folder_path}")
    print("="*60)
    
    # Load data from folder
    print("\nLoading images...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        folder_path,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"Images found: {test_gen.n}")
    print(f"Classes: {list(test_gen.class_indices.keys())}")
    
    # Create model
    print("\nCreating model...")
    model, base_model = create_efficientnet_model(num_classes=10)
    base_model.trainable = True
    
    # Load weights
    weights_file = 'models/efficientnet_weights.h5'
    if not os.path.exists(weights_file):
        print(f"❌ Weights file not found: {weights_file}")
        return
    
    print(f"Loading weights from: {weights_file}")
    model.load_weights(weights_file)
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING")
    print("="*60)
    
    loss, accuracy = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Loss:     {loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed predictions
    print("\nGenerating predictions...")
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
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        print(f"{class_name:30s}: {class_correct:3d}/{class_total:3d} = {class_acc:.4f} ({class_acc*100:.2f}%)")
    
    return accuracy

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_on_folder.py <folder_path>")
        print("\nExamples:")
        print("  python test_on_folder.py dataset_proper_split/test")
        print("  python test_on_folder.py dataset_proper_split/val")
        print("  python test_on_folder.py my_custom_images")
        print("\nNote: Folder should contain subdirectories for each class")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    test_on_folder(folder_path)
