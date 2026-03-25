"""
Simple dataset utilities for Brain Tumor Detection
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class BrainTumorDataset(Dataset):
    """Simple PyTorch Dataset for brain tumor images"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Get all class folders
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Collect all image paths
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((
                            os.path.join(class_path, img_name),
                            self.class_to_idx[class_name]
                        ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_class_names(root_dir):
    """Get sorted list of class names"""
    return sorted(os.listdir(root_dir))