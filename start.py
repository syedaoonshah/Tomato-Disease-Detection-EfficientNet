"""
Quick Start Script for Tomato Disease Classification
This script provides a simple interface to train and evaluate models.
"""

import sys
import os

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def check_dataset():
    """Check if dataset exists."""
    train_path = "dataset/train"
    valid_path = "dataset/valid"
    
    if not os.path.exists(train_path) or not os.path.exists(valid_path):
        print("ERROR: Dataset not found!")
        print(f"Expected folders:")
        print(f"  - {train_path}")
        print(f"  - {valid_path}")
        return False
    
    # Count classes
    train_classes = len([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    valid_classes = len([d for d in os.listdir(valid_path) if os.path.isdir(os.path.join(valid_path, d))])
    
    print(f"✓ Dataset found:")
    print(f"  Training classes: {train_classes}")
    print(f"  Validation classes: {valid_classes}")
    
    if train_classes != 11 or valid_classes != 11:
        print("WARNING: Expected 11 classes in each folder!")
        return False
    
    return True

def check_models():
    """Check if trained models exist."""
    models = {
        'ResNet50': 'models/resnet50_model.h5',
        'MobileNetV2': 'models/mobilenetv2_model.h5',
        'Ensemble': 'models/ensemble_model.h5'
    }
    
    existing = []
    missing = []
    
    for name, path in models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            existing.append(f"{name} ({size_mb:.1f} MB)")
        else:
            missing.append(name)
    
    if existing:
        print("✓ Trained models found:")
        for model in existing:
            print(f"  - {model}")
    
    if missing:
        print("\n✗ Missing models:")
        for model in missing:
            print(f"  - {model}")
    
    return len(existing), len(missing)

def main():
    """Main menu."""
    print_header("TOMATO LEAF DISEASE CLASSIFICATION")
    
    print("Project Structure:")
    print("  Source code: src/")
    print("  Dataset: dataset/train, dataset/valid")
    print("  Models: models/")
    print("  Results: results/")
    
    # Check dataset
    print_header("DATASET CHECK")
    if not check_dataset():
        print("\nPlease ensure your dataset is in the correct location.")
        print("See README.md for dataset structure requirements.")
        return
    
    # Check existing models
    print_header("MODEL CHECK")
    existing, missing = check_models()
    
    # Menu
    print_header("WHAT WOULD YOU LIKE TO DO?")
    print("1. Train all models (ResNet50, MobileNetV2, Ensemble)")
    print("2. Train ResNet50 only")
    print("3. Train MobileNetV2 only")
    print("4. Train Ensemble only")
    print("5. Evaluate trained models")
    print("6. View dataset information")
    print("7. Check GPU availability")
    print("8. Exit")
    
    choice = input("\nEnter your choice (1-8): ").strip()
    
    if choice == '1':
        print_header("TRAINING ALL MODELS")
        print("This will train ResNet50, MobileNetV2, and Ensemble models.")
        print("Expected time: 6-8 hours with GPU, 24-32 hours with CPU")
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm == 'y':
            os.system('python src/train.py')
    
    elif choice == '2':
        print_header("TRAINING RESNET50")
        print("Note: Use option 1 to train all models for best results")
        print("This option is for advanced users only")
    
    elif choice == '3':
        print_header("TRAINING MOBILENETV2")
        print("Note: Use option 1 to train all models for best results")
        print("This option is for advanced users only")
    
    elif choice == '4':
        print_header("TRAINING ENSEMBLE")
        print("Note: Ensemble requires pre-trained feature extractors")
        print("Use option 1 to train all models properly")
    
    elif choice == '5':
        print_header("EVALUATING MODELS")
        if missing == 3:
            print("ERROR: No trained models found!")
            print("Please train models first (option 1)")
        else:
            os.system('python src/evaluate.py')
    
    elif choice == '6':
        print_header("DATASET INFORMATION")
        os.system('python src/data_preprocessing.py')
    
    elif choice == '7':
        print_header("GPU CHECK")
        os.system('python -c "import tensorflow as tf; print(\'TensorFlow:\', tf.__version__); gpus = tf.config.list_physical_devices(\'GPU\'); print(\'GPU devices:\', gpus); print(\'GPU available:\', len(gpus) > 0)"')
        print("\nSee GPU_SETUP.md for GPU configuration on Windows")
    
    elif choice == '8':
        print("\nGoodbye!")
        return
    
    else:
        print("\nInvalid choice. Please run again and select 1-8.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please check the error message and try again.")
