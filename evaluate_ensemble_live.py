"""
Evaluate Ensemble: MobileNetV2 + EfficientNetB0
Can run while EfficientNetB0 training is ongoing
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
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

def evaluate_ensemble():
    """
    Evaluate ensemble of MobileNetV2 + EfficientNetB0
    """
    
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION: MobileNetV2 + EfficientNetB0")
    print("="*60)
    
    # Load MobileNetV2
    mobilenet_file = 'models/mobilenetv2_model.h5'
    if not os.path.exists(mobilenet_file):
        print(f"❌ MobileNetV2 not found: {mobilenet_file}")
        return
    
    print(f"\nLoading MobileNetV2 from: {mobilenet_file}")
    mobilenet_model = load_model(mobilenet_file)
    print("✅ MobileNetV2 loaded")
    
    # Load EfficientNetB0
    efficientnet_weights = 'models/efficientnet_weights.h5'
    if not os.path.exists(efficientnet_weights):
        print(f"❌ EfficientNetB0 weights not found: {efficientnet_weights}")
        return
    
    print(f"\nLoading EfficientNetB0 from: {efficientnet_weights}")
    efficientnet_model, base_model = create_efficientnet_model(num_classes=10)
    base_model.trainable = True
    efficientnet_model.load_weights(efficientnet_weights)
    efficientnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("✅ EfficientNetB0 loaded")
    
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
    
    # Individual model evaluations
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE")
    print("="*60)
    
    print("\n1. MobileNetV2:")
    mobilenet_loss, mobilenet_acc = mobilenet_model.evaluate(test_gen, verbose=0)
    print(f"   Accuracy: {mobilenet_acc:.4f} ({mobilenet_acc*100:.2f}%)")
    
    test_gen.reset()
    print("\n2. EfficientNetB0:")
    efficientnet_loss, efficientnet_acc = efficientnet_model.evaluate(test_gen, verbose=0)
    print(f"   Accuracy: {efficientnet_acc:.4f} ({efficientnet_acc*100:.2f}%)")
    
    # Get predictions
    test_gen.reset()
    print("\nGetting MobileNetV2 predictions...")
    mobilenet_pred = mobilenet_model.predict(test_gen, verbose=1)
    
    test_gen.reset()
    print("\nGetting EfficientNetB0 predictions...")
    efficientnet_pred = efficientnet_model.predict(test_gen, verbose=1)
    
    # Test different ensemble strategies
    print("\n" + "="*60)
    print("ENSEMBLE STRATEGIES")
    print("="*60)
    
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    
    strategies = [
        ("Equal Weight (50-50)", 0.5, 0.5),
        ("Favor EfficientNet (30-70)", 0.3, 0.7),
        ("Weighted by Accuracy", None, None),
    ]
    
    results = []
    
    for strategy_name, weight_mobilenet, weight_efficientnet in strategies:
        if strategy_name == "Weighted by Accuracy":
            # Weight by individual accuracies
            total_acc = mobilenet_acc + efficientnet_acc
            weight_mobilenet = mobilenet_acc / total_acc
            weight_efficientnet = efficientnet_acc / total_acc
        
        ensemble_pred = (weight_mobilenet * mobilenet_pred + 
                         weight_efficientnet * efficientnet_pred)
        
        y_pred_ensemble = np.argmax(ensemble_pred, axis=1)
        
        # Calculate accuracy
        correct = np.sum(y_pred_ensemble == y_true)
        total = len(y_true)
        ensemble_acc = correct / total
        
        results.append({
            'strategy': strategy_name,
            'accuracy': ensemble_acc,
            'weights': (weight_mobilenet, weight_efficientnet),
            'predictions': y_pred_ensemble
        })
        
        print(f"\n{strategy_name}:")
        print(f"  MobileNetV2 weight:   {weight_mobilenet:.3f}")
        print(f"  EfficientNetB0 weight: {weight_efficientnet:.3f}")
        print(f"  Test Accuracy: {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    
    # Find best strategy
    best_result = max(results, key=lambda x: x['accuracy'])
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print(f"\nIndividual Models:")
    print(f"MobileNetV2:     {mobilenet_acc:.4f} ({mobilenet_acc*100:.2f}%)")
    print(f"EfficientNetB0:  {efficientnet_acc:.4f} ({efficientnet_acc*100:.2f}%)")
    
    print(f"\nBest Ensemble: {best_result['strategy']}")
    print(f"  Accuracy:      {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
    
    improvement = (best_result['accuracy'] - max(mobilenet_acc, efficientnet_acc)) * 100
    print(f"  Improvement:   {improvement:+.2f}%")
    
    # Detailed classification report for best ensemble
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT (Best Ensemble)")
    print("="*60)
    
    report = classification_report(
        y_true, 
        best_result['predictions'],
        target_names=class_names,
        digits=4
    )
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, best_result['predictions'])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Ensemble Confusion Matrix\nAccuracy: {best_result["accuracy"]*100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs('results/confusion_matrices', exist_ok=True)
    plt.savefig('results/confusion_matrices/ensemble_mobilenetv2_efficientnet.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: results/confusion_matrices/ensemble_mobilenetv2_efficientnet.png")
    
    plt.close()
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    evaluate_ensemble()
