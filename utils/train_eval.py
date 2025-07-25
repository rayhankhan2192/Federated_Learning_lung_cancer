import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import os
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        def __call__(self, score: float) -> bool:
            if self.best_score in None:
                self.best_score = score
            elif (self.mode == 'min'):
                if score < self.best_score - self.min_delta:
                    self.best_score = score
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                if score > self.best_score + self.min_delta:
                    self.best_score = score
                    self.counter = 0
                else:
                    self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True  
            return self.early_stop
        
class ModelMatrics:
    """Class to compute and store model metrics"""
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = ['Bengin cases', 'Malignant cases', 'Normal cases']

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_probs: Optional[np.ndarray] = None) -> Dict:
        """Calculate accuracy, precision, recall, F1 score, and AUC"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }

        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name.lower()}_precision'] = precision[i] if i < len(precision) else 0
            metrics[f'{class_name.lower()}_recall'] = recall[i] if i < len(recall) else 0
            metrics[f'{class_name.lower()}_f1'] = f1[i] if i < len(f1) else 0
        

        if len(np.uniqe(y_true)) == 3:
            cm = confusion_matrix(y_true, y_pred)
            for i, class_name in enumerate(self.class_names):
                if i < cm.shape[0]:
                    tp = cm[i, i]
                    fn = np.sum(cm[i, :]) - tp
                    fp = np.sum(cm[:, i]) - tp
                    tn = np.sum(cm) - (tp + fn + fp)

                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    metrics[f'{class_name.lower()}_sensitivity'] = sensitivity  
                    metrics[f'{class_name.lower()}_specificity'] = specificity
        
        # AUC-ROC if probabilities are provided
        if y_probs is not None:
            try:
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true, y_probs[:, 1])
                    metrics['auc_roc'] = auc
                else:
                    auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
                    metrics['auc_roc_macro'] = auc
            except ValueError:
                logger.warning("Could not calculate AUC-ROC score")
        
        return metrics