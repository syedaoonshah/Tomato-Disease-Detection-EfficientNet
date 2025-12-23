"""
Model Evaluation Script
Generate:
- Confusion matrices
- Classification reports (Precision, Recall, F1-Score)
- Accuracy metrics
- Training/validation plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from tensorflow.keras.models import load_model
import tensorflow as tf
import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_preprocessing import create_test_generator

def get_class_names_from_generator(generator):
    """Extract class names from generator."""
    # Get class names from indices (reverse mapping)
    class_names = [''] * len(generator.class_indices)
    for name, idx in generator.class_indices.items():
        class_names[idx] = name
    return class_names

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    plt.ylabel('Actual Value', fontsize=12)
    plt.xlabel('Predicted Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrices/{model_name}_cm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: results/confusion_matrices/{model_name}_cm.png")

def evaluate_model(model_path, test_gen, model_name):
    """Evaluate a single model."""
    
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Get class names
    class_names = get_class_names_from_generator(test_gen)
    
    # Reset generator
    test_gen.reset()
    
    # Predictions
    print("Making predictions...")
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
    print(f"\n{model_name} Classification Report:")
    print(report)
    
    # Save report
    with open(f'results/classification_reports/{model_name}_report.txt', 'w') as f:
        f.write(f"{model_name} Evaluation Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Classification Report:\n")
        f.write("="*60 + "\n")
        f.write(report)
        
        # Add per-class accuracy
        f.write("\n\nPer-Class Accuracy:\n")
        f.write("="*60 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        for i, class_name in enumerate(class_names):
            class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            f.write(f"{class_name:40s}: {class_acc:.4f} ({class_acc*100:.2f}%)\n")
    
    print(f"Report saved: results/classification_reports/{model_name}_report.txt")
    
    # Confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, model_name)
    
    # Find misclassifications
    misclassified = np.where(y_true != y_pred)[0]
    print(f"\nTotal misclassifications: {len(misclassified)}")
    
    if len(misclassified) > 0 and len(misclassified) <= 20:
        print("\nMisclassified samples:")
        for idx in misclassified[:20]:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = predictions[idx][y_pred[idx]]
            print(f"  Sample {idx}: True={true_class}, Predicted={pred_class} (confidence: {confidence:.4f})")
    
    return accuracy, y_true, y_pred, class_names

def plot_comparison(results):
    """Plot comparison of model accuracies."""
    
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylim([0.85, 1.0])
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison plot saved: results/model_comparison.png")

def main():
    """Main evaluation function."""
    
    print("\n" + "="*60)
    print("TOMATO LEAF DISEASE CLASSIFICATION - EVALUATION")
    print("="*60)
    
    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU available: {gpus}")
        except RuntimeError as e:
            print(e)
    
    # Load test data
    print("\nLoading test data...")
    test_gen = create_test_generator()
    
    print(f"\nTest Information:")
    print(f"Test samples: {test_gen.samples}")
    print(f"Number of classes: {len(test_gen.class_indices)}")
    print(f"Batch size: {test_gen.batch_size}")
    
    # Check if model files exist
    models = {
        'ResNet50': 'models/resnet50_model.h5',
        'MobileNetV2': 'models/mobilenetv2_model.h5',
        'Ensemble': 'models/ensemble_model.h5'
    }
    
    # Filter only existing models
    existing_models = {}
    for name, path in models.items():
        if os.path.exists(path):
            existing_models[name] = path
        else:
            print(f"\nWarning: {name} model not found at {path}")
    
    if not existing_models:
        print("\nError: No trained models found. Please run 'python src/train.py' first.")
        return
    
    # Evaluate models
    results = {}
    for name, path in existing_models.items():
        accuracy, y_true, y_pred, class_names = evaluate_model(path, test_gen, name)
        results[name] = {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Reset generator for next model
        test_gen.reset()
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)
    print("\nTest Accuracies:")
    for name, res in results.items():
        acc = res['accuracy']
        print(f"{name:15s}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Plot comparison
    if len(results) > 1:
        plot_comparison(results)
    
    print("\nGenerated files:")
    print("  Confusion Matrices: results/confusion_matrices/")
    print("  Classification Reports: results/classification_reports/")
    print("  Model Comparison: results/model_comparison.png")
    
    # Expected results check
    if 'Ensemble' in results:
        ensemble_acc = results['Ensemble']['accuracy']
        print(f"\n{'='*60}")
        print("Expected vs Actual Results:")
        print(f"{'='*60}")
        print(f"Expected Ensemble Accuracy: 0.9991 (99.91%)")
        print(f"Actual Ensemble Accuracy:   {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
        
        if ensemble_acc >= 0.99:
            print("\n✓ Excellent! Achieved target accuracy (>99%)")
        elif ensemble_acc >= 0.95:
            print("\n✓ Good accuracy achieved (>95%)")
        else:
            print("\n⚠ Accuracy below expected. Consider:")
            print("  - Training for more epochs")
            print("  - Checking data quality")
            print("  - Verifying model architecture")

if __name__ == '__main__':
    main()
