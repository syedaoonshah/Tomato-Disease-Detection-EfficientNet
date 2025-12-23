"""
Create proper 80-10-10 split from your existing 11,000 images
Combines train (10,000) + valid (1,000) = 11,000 total
Then splits into 8,800 / 1,100 / 1,100 to match the paper exactly
"""
import os
import shutil
import random
from pathlib import Path

def create_8800_1100_1100_split():
    """
    Combine train (10,000) + valid (1,000) = 11,000 total
    Then split into 8,800 / 1,100 / 1,100
    """
    
    random.seed(42)
    
    # Source directories
    train_source = 'dataset/train'
    valid_source = 'dataset/val'
    
    # Output directories
    output_base = 'dataset_proper_split'
    
    # Expected classes
    classes = [
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
    
    print("="*70)
    print("CREATING PROPER 80-10-10 SPLIT (8,800 / 1,100 / 1,100)")
    print("="*70)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(f'{output_base}/{split}/{cls}', exist_ok=True)
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for cls in classes:
        print(f"\nProcessing: {cls}")
        
        # Collect all images from both train and valid folders
        all_images = []
        
        # From train folder (1,000 images)
        train_cls_path = os.path.join(train_source, cls)
        if os.path.exists(train_cls_path):
            train_imgs = [(os.path.join(train_cls_path, f), f) 
                         for f in os.listdir(train_cls_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(train_imgs)
        
        # From valid folder (~100 images per class)
        valid_cls_path = os.path.join(valid_source, cls)
        if os.path.exists(valid_cls_path):
            valid_imgs = [(os.path.join(valid_cls_path, f), f) 
                         for f in os.listdir(valid_cls_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.extend(valid_imgs)
        
        total_images = len(all_images)
        print(f"  Total images found: {total_images}")
        
        if total_images != 1100:
            print(f"  ⚠️  WARNING: Expected 1,100, found {total_images}")
        
        # Shuffle
        random.shuffle(all_images)
        
        # Split: 80% train, 10% val, 10% test
        n = len(all_images)
        train_end = int(n * 0.8)      # 880
        val_end = train_end + int(n * 0.1)  # 990
        
        train_split = all_images[:train_end]
        val_split = all_images[train_end:val_end]
        test_split = all_images[val_end:]
        
        # Copy files
        for src_path, filename in train_split:
            dst = f'{output_base}/train/{cls}/{filename}'
            shutil.copy2(src_path, dst)
        
        for src_path, filename in val_split:
            dst = f'{output_base}/val/{cls}/{filename}'
            shutil.copy2(src_path, dst)
        
        for src_path, filename in test_split:
            dst = f'{output_base}/test/{cls}/{filename}'
            shutil.copy2(src_path, dst)
        
        total_train += len(train_split)
        total_val += len(val_split)
        total_test += len(test_split)
        
        print(f"  → Train: {len(train_split):4d} | Val: {len(val_split):3d} | Test: {len(test_split):3d}")
    
    print("\n" + "="*70)
    print("SPLIT COMPLETE!")
    print("="*70)
    print(f"\nFinal Distribution:")
    print(f"  Training:   {total_train:,} images (target: 8,800)")
    print(f"  Validation: {total_val:,} images (target: 1,100)")
    print(f"  Testing:    {total_test:,} images (target: 1,100)")
    print(f"  TOTAL:      {total_train + total_val + total_test:,} images")
    
    if abs(total_train - 8800) < 50:
        print(f"\n✅ Training set size is correct!")
    if abs(total_val - 1100) < 20:
        print(f"✅ Validation set size is correct!")
    if abs(total_test - 1100) < 20:
        print(f"✅ Test set size is correct!")

if __name__ == '__main__':
    create_8800_1100_1100_split()
