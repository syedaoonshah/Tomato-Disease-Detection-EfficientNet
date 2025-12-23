"""
Predict on individual images in a folder (no subdirectories required)
Usage: python predict_images.py <folder_path>
Example: python predict_images.py custom/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

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

# Class names (must match training order)
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def load_and_preprocess_image(img_path):
    """Load and preprocess a single image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_images(folder_path):
    """Predict classes for all images in a folder."""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    print("\n" + "="*60)
    print(f"PREDICTING ON IMAGES IN: {folder_path}")
    print("="*60)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f.lower())[1] in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        print(f"Looking for: {', '.join(image_extensions)}")
        return
    
    print(f"Found {len(image_files)} images")
    
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
    
    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    
    # Predict each image
    predictions_summary = {class_name: 0 for class_name in CLASS_NAMES}
    
    for img_file in sorted(image_files):
        img_path = os.path.join(folder_path, img_file)
        
        try:
            # Load and preprocess
            img_array = load_and_preprocess_image(img_path)
            
            # Predict
            pred = model.predict(img_array, verbose=0)
            pred_class_idx = np.argmax(pred[0])
            pred_class = CLASS_NAMES[pred_class_idx]
            confidence = pred[0][pred_class_idx] * 100
            
            # Update summary
            predictions_summary[pred_class] += 1
            
            # Print result
            print(f"{img_file:30s} → {pred_class:45s} ({confidence:.2f}%)")
            
        except Exception as e:
            print(f"{img_file:30s} → ERROR: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print("\nPredictions by class:")
    for class_name, count in sorted(predictions_summary.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            percentage = (count / len(image_files)) * 100
            print(f"  {class_name:45s}: {count:3d} ({percentage:.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict_images.py <folder_path>")
        print("\nExamples:")
        print("  python predict_images.py custom/")
        print("  python predict_images.py my_images/")
        print("\nNote: Images can be directly in the folder (no subdirectories needed)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    predict_images(folder_path)
