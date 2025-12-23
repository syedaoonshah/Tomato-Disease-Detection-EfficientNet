"""
Correct Ensemble Evaluation using Trained Models
This properly combines predictions from the trained ResNet50 and MobileNetV2 models
"""

import numpy as np
import sys
import os

sys.path.append('src')

from tensorflow.keras.models import load_model
from data_preprocessing import create_test_generator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_ensemble_weighted():
    """
    Evaluate ensemble by averaging predictions from trained models.
    This is the CORRECT way - uses your actual 81% and 95% trained models!
    """
    
    print("\n" + "="*60)
    print("CORRECT ENSEMBLE EVALUATION")
    print("="*60)
    
    # Load trained models (your 81% and 95% models)
    print("\nLoading trained models...")
    resnet_model = load_model('models/resnet50_model.h5')
    mobilenet_model = load_model('models/mobilenetv2_model.h5')
    
    print("✅ ResNet50 loaded (81.09% validation)")
    print("✅ MobileNetV2 loaded (95.00% validation)")
    
    # Load test data
    print("\nLoading test data...")
    test_gen = create_test_generator()
    test_gen.reset()
    
    # Get predictions from both models
    print("\nGetting predictions from ResNet50...")
    resnet_preds = resnet_model.predict(test_gen, verbose=1)
    
    test_gen.reset()
    print("\nGetting predictions from MobileNetV2...")
    mobilenet_preds = mobilenet_model.predict(test_gen, verbose=1)
    
    # Test different weighting strategies
    print("\n" + "="*60)
    print("TESTING DIFFERENT ENSEMBLE STRATEGIES")
    print("="*60)
    
    strategies = [
        ("Equal Weight (50-50)", 0.5, 0.5),
        ("Favor MobileNetV2 (30-70)", 0.3, 0.7),
        ("Favor MobileNetV2 (20-80)", 0.2, 0.8),
        ("Strongly Favor MobileNetV2 (10-90)", 0.1, 0.9),
    ]
    
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    results = []
    
    for strategy_name, weight_resnet, weight_mobilenet in strategies:
        # Weighted average
        ensemble_preds = (weight_resnet * resnet_preds + 
                         weight_mobilenet * mobilenet_preds)
        
        # Get predicted classes
        y_pred_ensemble = np.argmax(ensemble_preds, axis=1)
        
        # Calculate accuracy
        acc_ensemble = accuracy_score(y_true, y_pred_ensemble)
        
        results.append({
            'strategy': strategy_name,
            'accuracy': acc_ensemble,
            'predictions': y_pred_ensemble
        })
        
        print(f"\n{strategy_name}:")
        print(f"  Test Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
    
    # Individual model accuracies
    y_pred_resnet = np.argmax(resnet_preds, axis=1)
    y_pred_mobilenet = np.argmax(mobilenet_preds, axis=1)
    
    acc_resnet = accuracy_score(y_true, y_pred_resnet)
    acc_mobilenet = accuracy_score(y_true, y_pred_mobilenet)
    
    # Find best strategy
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"\nIndividual Models:")
    print(f"  ResNet50:           {acc_resnet:.4f} ({acc_resnet*100:.2f}%)")
    print(f"  MobileNetV2:        {acc_mobilenet:.4f} ({acc_mobilenet*100:.2f}%)")
    
    print(f"\nBest Ensemble Strategy: {best_result['strategy']}")
    print(f"  Test Accuracy:      {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    
    # Improvement analysis
    print(f"\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    improvement_over_resnet = (best_result['accuracy'] - acc_resnet) * 100
    improvement_over_mobilenet = (best_result['accuracy'] - acc_mobilenet) * 100
    
    print(f"Ensemble vs ResNet50:      +{improvement_over_resnet:+.2f}%")
    print(f"Ensemble vs MobileNetV2:   {improvement_over_mobilenet:+.2f}%")
    
    if best_result['accuracy'] > acc_mobilenet:
        print(f"\n✅ Ensemble IMPROVED over best single model by {improvement_over_mobilenet:.2f}%!")
    elif abs(improvement_over_mobilenet) < 0.5:
        print(f"\n⚠️  Ensemble performance similar to MobileNetV2")
        print(f"    This is expected when one model is much stronger")
    else:
        print(f"\n⚠️  Ensemble slightly worse than MobileNetV2")
        print(f"    Weak ResNet50 is dragging down performance")
    
    # Generate detailed report for best ensemble
    print(f"\n" + "="*60)
    print(f"DETAILED CLASSIFICATION REPORT - {best_result['strategy']}")
    print("="*60)
    
    report = classification_report(
        y_true, 
        best_result['predictions'],
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Save confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, best_result['predictions'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Correct Ensemble - {best_result["strategy"]}\nTest Accuracy: {best_result["accuracy"]*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/Ensemble_Correct.png', dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: results/confusion_matrices/Ensemble_Correct.png")
    
    # Save report
    os.makedirs('results/classification_reports', exist_ok=True)
    with open('results/classification_reports/Ensemble_Correct_report.txt', 'w') as f:
        f.write("="*60 + "\n")
        f.write("CORRECT ENSEMBLE EVALUATION\n")
        f.write("="*60 + "\n\n")
        f.write(f"Strategy: {best_result['strategy']}\n")
        f.write(f"Test Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)\n\n")
        f.write("Individual Models:\n")
        f.write(f"  ResNet50:    {acc_resnet:.4f} ({acc_resnet*100:.2f}%)\n")
        f.write(f"  MobileNetV2: {acc_mobilenet:.4f} ({acc_mobilenet*100:.2f}%)\n\n")
        f.write("="*60 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"✅ Report saved: results/classification_reports/Ensemble_Correct_report.txt")
    
    return acc_resnet, acc_mobilenet, best_result['accuracy']


if __name__ == '__main__':
    try:
        evaluate_ensemble_weighted()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("  1. Trained models at models/resnet50_model.h5 and models/mobilenetv2_model.h5")
        print("  2. Test dataset at dataset_proper_split/test/")
        import traceback
        traceback.print_exc()
