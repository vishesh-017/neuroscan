"""
Brain Tumor MRI Dataset Utilities
=================================
Provides clean PyTorch Dataset class, contour cropping, and image loading.
Designed for easy explanation and high code clarity.
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def crop_brain_contour(image_pil):
    """
    Finds the contours of the brain skull and crops out the excess black background.
    
    Why: In brain MRIs, 40-60% of the image can be empty black space.
         Cropping focuses the neural network on brain tissue and tumor anomalies.
    """
    image_np = np.array(image_pil)
    # Convert RGB to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    # Slight blur and threshold to find brain tissue
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray_blurred, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_pil
    
    # Largest contour corresponds to the brain skull
    c = max(contours, key=cv2.contourArea)
    
    # Get bounding coordinates
    ext_left = tuple(c[c[:, :, 0].argmin()][0])
    ext_right = tuple(c[c[:, :, 0].argmax()][0])
    ext_top = tuple(c[c[:, :, 1].argmin()][0])
    ext_bot = tuple(c[c[:, :, 1].argmax()][0])
    
    # Crop
    new_image_np = image_np[ext_top[1]:ext_bot[1], ext_left[0]:ext_right[0]]
    if new_image_np.size == 0:
        return image_pil
    
    return Image.fromarray(new_image_np)


class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for Brain Tumor MRI Images.
    Supports directory-based loading, optional contour cropping, and transforms.
    """
    def __init__(self, samples, transform=None, apply_crop=False):
        """
        Args:
            samples (list): List of tuples (image_path, label_idx)
            transform (callable, optional): PyTorch transform pipeline
            apply_crop (bool): Whether to crop brain contour
        """
        self.samples = samples
        self.transform = transform
        self.apply_crop = apply_crop

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.apply_crop:
            image = crop_brain_contour(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


def scan_dataset(root_dir):
    """
    Scans a folder structured as:
      root_dir/
        class_1/
          img1.jpg, ...
        class_2/
          ...
    Returns:
      samples: list of (path, label_idx)
      class_names: list of class name strings
      class_to_idx: dict mapping class name to integer index
    """
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    
    samples = []
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    for class_name in class_names:
        class_folder = os.path.join(root_dir, class_name)
        for fname in os.listdir(class_folder):
            if fname.lower().endswith(valid_exts):
                samples.append((os.path.join(class_folder, fname), class_to_idx[class_name]))
                
    return samples, class_names, class_to_idx