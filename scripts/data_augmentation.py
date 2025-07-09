# ==================== scripts/data_augmentation.py ====================
import albumentations as A
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

class AdvancedDataAugmentation:
    def __init__(self):
        self.augmentation = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.HueSaturationValue(p=1.0),
            ], p=0.8),
            
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(blur_limit=3, p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.RandomShadow(p=1.0),
                A.RandomFog(p=1.0),
                A.RandomSunFlare(p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.ElasticTransform(p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(p=1.0),
            ], p=0.2),
            
            A.RandomRotate90(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ])
    
    def augment_dataset(self, input_dir, output_dir, multiplier=3):
        """Augment existing dataset"""
        
        # Load original annotations
        with open(f"{input_dir}/annotations.json", 'r') as f:
            annotations = json.load(f)
        
        new_annotations = []
        
        for annotation in tqdm(annotations, desc="Augmenting dataset"):
            # Load original image
            img_path = f"{input_dir}/images/{annotation['filename']}"
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Add original annotation
            new_annotations.append(annotation)
            
            # Generate augmented versions
            for i in range(multiplier):
                augmented = self.augmentation(image=image)
                aug_image = augmented['image']
                
                # Save augmented image
                aug_filename = f"aug_{i}_{annotation['filename']}"
                aug_path = f"{output_dir}/images/{aug_filename}"
                
                os.makedirs(os.path.dirname(aug_path), exist_ok=True)
                cv2.imwrite(aug_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
                
                # Create new annotation
                new_annotation = annotation.copy()
                new_annotation['filename'] = aug_filename
                new_annotation['image_id'] = len(new_annotations)
                new_annotations.append(new_annotation)
        
        # Save new annotations
        with open(f"{output_dir}/annotations.json", 'w') as f:
            json.dump(new_annotations, f, indent=2)
        
        print(f"Dataset augmented: {len(annotations)} -> {len(new_annotations)} images")