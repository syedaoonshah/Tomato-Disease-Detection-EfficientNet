"""
Compare EfficientNetB0 vs Ensemble Model
Generate comparison graphs for GitHub
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_ensemble_model(num_classes=10):
    """Recreate ensemble model architecture"""
    
    # Load MobileNetV2
    mobilenet_full = load_model('models/mobilenetv2_model.h5')
    mobilenet_feature_layer = None
    for i in range(len(mobilenet_full.layers) - 1, -1, -1):
        layer = mobilenet_full.layers[i]
        if isinstance(layer, layers.Dense) and layer.activation.__name__ == 'relu':
            mobilenet_feature_layer = layer.output
            break
    if mobilenet_feature_layer is None:
        mobilenet_feature_layer = mobilenet_full.layers[-2].output
    
    mobilenet_base = Model(
        inputs=mobilenet_full.input,
        outputs=mobilenet_feature_layer,
        name='mobilenet_base'
    )
    mobilenet_base.trainable = False
    
    # Load EfficientNetB0
    efficientnet_full, _ = create_efficientnet_model(num_classes=10)
    efficientnet_full.load_weights('models/efficientnet_weights.h5')
    efficientnet_feature_layer = None
    for i in range(len(efficientnet_full.layers) - 1, -1, -1):
        layer = efficientnet_full.layers[i]
        if isinstance(layer, layers.Dense) and layer.activation.__name__ == 'relu':
            efficientnet_feature_layer = layer.output
            break
    if efficientnet_feature_layer is None:
        efficientnet_feature_layer = efficientnet_full.layers[-2].output
    
    efficientnet_base = Model(
        inputs=efficientnet_full.input,
        outputs=efficientnet_feature_layer,
        name='efficientnet_base'
    )
    efficientnet_base.trainable = False
    
    # Create ensemble
    input_layer = layers.Input(shape=(224, 224, 3), name='ensemble_input')
    mobilenet_features = mobilenet_base(input_layer)
    efficientnet_features = efficientnet_base(input_layer)
    concatenated = layers.Concatenate(name='concat_features')([mobilenet_features, efficientnet_features])
    
    x = layers.Dense(256, activation='relu', name='meta_dense1')(concatenated)
    x = layers.BatchNormalization(name='meta_bn1')(x)
    x = layers.Dropout(0.4, name='meta_dropout1')(x)
    x = layers.Dense(128, activation='relu', name='meta_dense2')(x)
    x = layers.BatchNormalization(name='meta_bn2')(x)
    x = layers.Dropout(0.3, name='meta_dropout2')(x)
    output = layers.Dense(num_classes, activation='softmax', name='ensemble_output')(x)
    
    ensemble_model = Model(inputs=input_layer, outputs=output, name='ensemble_model')
    return ensemble_model

def evaluate_models():
    """Evaluate both models and create comparison"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: EfficientNetB0 vs Ensemble")
    print("="*80)
    
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
    
    class_names = list(test_gen.class_indices.keys())
    y_true = test_gen.classes
    
    print(f"Test samples: {test_gen.n}")
    print(f"Classes: {len(class_names)}")
    
    # 1. Evaluate EfficientNetB0
    print("\n" + "="*80)
    print("EVALUATING EFFICIENTNETB0")
    print("="*80)
    
    efficientnet_model, base_model = create_efficientnet_model(num_classes=10)
    base_model.trainable = True
    efficientnet_model.load_weights('models/efficientnet_weights.h5')
    efficientnet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    test_gen.reset()
    eff_loss, eff_acc = efficientnet_model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {eff_acc:.4f} ({eff_acc*100:.2f}%)")
    
    test_gen.reset()
    eff_pred = efficientnet_model.predict(test_gen, verbose=0)
    eff_pred_classes = np.argmax(eff_pred, axis=1)
    
    # 2. Evaluate Ensemble
    print("\n" + "="*80)
    print("EVALUATING ENSEMBLE MODEL")
    print("="*80)
    
    ensemble_model = create_ensemble_model(num_classes=10)
    ensemble_model.load_weights('models/ensemble_efficientnet_best_weights.h5')
    ensemble_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    test_gen.reset()
    ens_loss, ens_acc = ensemble_model.evaluate(test_gen, verbose=0)
    print(f"Test Accuracy: {ens_acc:.4f} ({ens_acc*100:.2f}%)")
    
    test_gen.reset()
    ens_pred = ensemble_model.predict(test_gen, verbose=0)
    ens_pred_classes = np.argmax(ens_pred, axis=1)
    
    # 3. Per-class comparison
    print("\n" + "="*80)
    print("PER-CLASS ACCURACY COMPARISON")
    print("="*80)
    
    eff_cm = confusion_matrix(y_true, eff_pred_classes)
    ens_cm = confusion_matrix(y_true, ens_pred_classes)
    
    per_class_results = []
    for i, class_name in enumerate(class_names):
        eff_class_acc = eff_cm[i, i] / eff_cm[i].sum() if eff_cm[i].sum() > 0 else 0
        ens_class_acc = ens_cm[i, i] / ens_cm[i].sum() if ens_cm[i].sum() > 0 else 0
        
        per_class_results.append({
            'Class': class_name.replace('Tomato___', '').replace('_', ' '),
            'EfficientNetB0': eff_class_acc * 100,
            'Ensemble': ens_class_acc * 100
        })
        
        print(f"{class_name:40s}: EfficientNet: {eff_class_acc*100:5.2f}%  |  Ensemble: {ens_class_acc*100:5.2f}%")
    
    df_per_class = pd.DataFrame(per_class_results)
    
    # 4. Generate visualizations
    print("\n" + "="*80)
    print("GENERATING COMPARISON GRAPHS")
    print("="*80)
    
    os.makedirs('results/comparisons', exist_ok=True)
    
    # Graph 1: Overall accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['MobileNetV2', 'EfficientNetB0', 'Ensemble']
    accuracies = [92.45, eff_acc*100, ens_acc*100]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([90, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/comparisons/model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/comparisons/model_accuracy_comparison.png")
    plt.close()
    
    # Graph 2: Per-class accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(class_names))
    width = 0.35
    
    class_labels = [name.replace('Tomato___', '').replace('_', ' ') for name in class_names]
    eff_class_accs = [eff_cm[i, i] / eff_cm[i].sum() * 100 for i in range(len(class_names))]
    ens_class_accs = [ens_cm[i, i] / ens_cm[i].sum() * 100 for i in range(len(class_names))]
    
    bars1 = ax.bar(x - width/2, eff_class_accs, width, label='EfficientNetB0', color='#2ecc71', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, ens_class_accs, width, label='Ensemble', color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy: EfficientNetB0 vs Ensemble', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([90, 101])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/comparisons/per_class_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/comparisons/per_class_comparison.png")
    plt.close()
    
    # Graph 3: Side-by-side confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    
    # EfficientNetB0 confusion matrix
    sns.heatmap(eff_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[0],
                cbar_kws={'label': 'Count'})
    axes[0].set_title(f'EfficientNetB0 Confusion Matrix\nAccuracy: {eff_acc*100:.2f}%', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    # Ensemble confusion matrix
    sns.heatmap(ens_cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_labels, yticklabels=class_labels, ax=axes[1],
                cbar_kws={'label': 'Count'})
    axes[1].set_title(f'Ensemble Model Confusion Matrix\nAccuracy: {ens_acc*100:.2f}%', 
                     fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/comparisons/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/comparisons/confusion_matrices_comparison.png")
    plt.close()
    
    # Graph 4: Difference heatmap (where ensemble improves/degrades)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate per-class improvement
    improvement = []
    for i in range(len(class_names)):
        eff_acc_i = eff_cm[i, i] / eff_cm[i].sum() * 100
        ens_acc_i = ens_cm[i, i] / ens_cm[i].sum() * 100
        improvement.append(ens_acc_i - eff_acc_i)
    
    # Create single column heatmap
    improvement_matrix = np.array(improvement).reshape(-1, 1)
    
    sns.heatmap(improvement_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                yticklabels=class_labels, xticklabels=['Improvement (%)'],
                cbar_kws={'label': 'Accuracy Change (%)'},
                vmin=-5, vmax=5, ax=ax)
    
    ax.set_title('Ensemble vs EfficientNetB0: Per-Class Improvement', 
                fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('results/comparisons/improvement_heatmap.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/comparisons/improvement_heatmap.png")
    plt.close()
    
    # 5. Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\n{'Model':<20} {'Test Accuracy':<15} {'Test Loss':<15}")
    print("-" * 50)
    print(f"{'MobileNetV2':<20} {'92.45%':<15} {'-':<15}")
    print(f"{'EfficientNetB0':<20} {f'{eff_acc*100:.2f}%':<15} {f'{eff_loss:.4f}':<15}")
    print(f"{'Ensemble':<20} {f'{ens_acc*100:.2f}%':<15} {f'{ens_loss:.4f}':<15}")
    
    improvement_pct = (ens_acc - eff_acc) * 100
    print(f"\n{'='*80}")
    if improvement_pct > 0:
        print(f"✅ Ensemble IMPROVED over EfficientNetB0 by {improvement_pct:.2f}%")
    elif improvement_pct < 0:
        print(f"⚠️  Ensemble is {abs(improvement_pct):.2f}% below EfficientNetB0")
    else:
        print(f"✅ Ensemble MATCHES EfficientNetB0 performance")
    
    print(f"\n📊 All comparison graphs saved to: results/comparisons/")
    print(f"   - model_accuracy_comparison.png")
    print(f"   - per_class_comparison.png")
    print(f"   - confusion_matrices_comparison.png")
    print(f"   - improvement_heatmap.png")
    print("="*80)
    
    return {
        'efficientnet': {'accuracy': eff_acc, 'loss': eff_loss},
        'ensemble': {'accuracy': ens_acc, 'loss': ens_loss}
    }

if __name__ == '__main__':
    results = evaluate_models()
