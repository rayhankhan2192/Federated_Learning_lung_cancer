import os
import torch
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CTScaneDataset(Dataset):

    def __init__(self, data_dir: str, transform=None, subset: str = 'train'):
        """
        Args:
            data_dir: Root directory containing class folders
            transform: Albumentations transform pipeline
            subset: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.subset = subset

        # Class mapping
        self.class_to_idx = {'Normal case': 0, 'Benign case': 1, 'Malignant case': 2}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Load all image paths and labels
        self.samples = self._load_samples()
        
        # Calculate class weights for balanced training
        self.class_weights = self._calculate_class_weights()

        logger.info(f"Loaded {len(self.samples)} samples for {subset} set")
        self._print_class_distribution()


    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their corresponding labels"""
        samples = []
        
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_path) or class_name not in self.class_to_idx:
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, class_idx))
        
        return samples
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets"""
        class_counts = [0] * len(self.class_to_idx)
        for _, label in self.samples:
            class_counts[label] += 1
        
        total_samples = len(self.samples)
        weights = [total_samples / (len(self.class_to_idx) * count) for count in class_counts]
        return torch.FloatTensor(weights)
    
    