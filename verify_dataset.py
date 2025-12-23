"""
Dataset Verification Script - Run this FIRST
"""
import os
from pathlib import Path

def verify_dataset():
    """Verify dataset structure matches paper requirements."""
    
    print("\n" + "="*70)
    print("DATASET VERIFICATION FOR TOMATO LEAF DISEASE CLASSIFICATION")
    print("="*70)
    
    # Expected classes from the paper (exactly 10)
    EXPECTED_CLASSES = [
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites_Two-spotted_spider_mite',
        'Tomato___Target_Spot',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    # Check if dataset directories exist
    dirs_to_check = {
        'train': 'dataset/train',
        'validation': 'dataset/valid',
        'test': 'dataset/test' if os.path.exists('dataset/test') else None
    }
    
    print("\n📂 Checking directory structure...")
    for name, path in dirs_to_check.items():
        if path and os.path.exists(path):
            print(f"✅ {name:12s}: {path}")
        else:
            print(f"❌ {name:12s}: NOT FOUND")
    
    # Analyze training set
    train_path = 'dataset/train'
    if not os.path.exists(train_path):
        print(f"\n❌ ERROR: Training directory not found at {train_path}")
        return False
    
    classes = sorted([d for d in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, d))])
    
    print(f"\n📊 TRAINING SET ANALYSIS")
    print("="*70)
    print(f"Number of classes found: {len(classes)}")
    
    if len(classes) != 10:
        print(f"❌ ERROR: Expected 10 classes, found {len(classes)}")
    else:
        print(f"✅ Correct number of classes (10)")
    
    print(f"\n📁 Class Distribution:")
    print("-"*70)
    
    total_images = 0
    issues = []
    
    for i, cls in enumerate(classes, 1):
        cls_path = os.path.join(train_path, cls)
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        
        # Check if class is expected
        if cls in EXPECTED_CLASSES:
            status = "✅"
        else:
            status = "❌"
            issues.append(f"Unexpected class: {cls}")
        
        # Check if count is reasonable (should be ~800 for 80% split of 1000)
        expected_count = 800  # 80% of 1000
        if abs(count - expected_count) > 100:
            status = "⚠️"
            issues.append(f"{cls}: {count} images (expected ~{expected_count})")
        
        print(f"{status} {i:2d}. {cls:55s} : {count:5d} images")
    
    print("-"*70)
    print(f"Total training images: {total_images}")
    print(f"Expected: ~8,000 (80% of 10,000)")
    
    # Check for missing expected classes
    print(f"\n🔍 Checking for expected classes...")
    missing_classes = set(EXPECTED_CLASSES) - set(classes)
    if missing_classes:
        print(f"❌ Missing classes:")
        for cls in missing_classes:
            print(f"   - {cls}")
            issues.append(f"Missing class: {cls}")
    else:
        print(f"✅ All expected classes present")
    
    # Check for extra classes
    extra_classes = set(classes) - set(EXPECTED_CLASSES)
    if extra_classes:
        print(f"\n❌ Extra classes found (should be removed):")
        for cls in extra_classes:
            print(f"   - {cls}")
            issues.append(f"Extra class: {cls}")
    else:
        print(f"✅ No extra classes")
    
    # Check validation set
    if os.path.exists('dataset/valid'):
        val_classes = sorted([d for d in os.listdir('dataset/valid') 
                             if os.path.isdir(os.path.join('dataset/valid', d))])
        val_total = sum([len(os.listdir(os.path.join('dataset/valid', c))) 
                        for c in val_classes])
        
        print(f"\n📊 VALIDATION SET")
        print("="*70)
        print(f"Number of classes: {len(val_classes)}")
        print(f"Total images: {val_total}")
        print(f"Expected: ~2,000 (20% of 10,000)")
        
        if abs(val_total - 2000) > 200:
            issues.append(f"Validation set: {val_total} images (expected ~2,000)")
    
    # Final summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    
    if issues:
        print(f"\n❌ FOUND {len(issues)} ISSUE(S):")
        for issue in issues:
            print(f"   • {issue}")
        print(f"\n⚠️  Your dataset does NOT match the paper's requirements!")
        print(f"    Please fix these issues before training.")
        return False
    else:
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"   Your dataset matches the paper's requirements.")
        print(f"   You can proceed with training.")
        return True

if __name__ == '__main__':
    verify_dataset()
