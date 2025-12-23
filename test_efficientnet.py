"""
Quick test to verify EfficientNetB0 model can be created
Run this BEFORE starting 4-hour training!
"""

import sys
sys.path.append('src')

print("Testing EfficientNetB0 model creation...")

try:
    from model_efficientnet import create_efficientnet_model, create_efficientnet_feature_extractor
    print("✅ Import successful!")
    
    print("\nCreating full model...")
    model, base = create_efficientnet_model(num_classes=10)
    print(f"✅ Model created!")
    print(f"   Total params: {model.count_params():,}")
    
    print("\nCreating feature extractor...")
    extractor = create_efficientnet_feature_extractor()
    print(f"✅ Feature extractor created!")
    print(f"   Output shape: {extractor.output_shape}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\nReady to train! Run:")
    print("  wsl bash train_efficientnet_gpu_wsl.sh")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nFix the error before training!")
    import traceback
    traceback.print_exc()
