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


class ModelTrain:
    """Main Training Engine for the model"""

    def __init__(self, model: nn.modules, device: torch.device, save_dir: str = 'checkpoints', log_dir: str = 'logs'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        self.log_dir = log_dir

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.matrics_calculator = ModelMatrics()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.history = defaultdict(list)

    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, epoch: int) -> Dict:
        """Train for one epoch"""

        self.model.train()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} - Training')

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(data)  
            loss = criterion(outputs, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().detach().numpy())

            # Update progress bar
            progress_bar.set_postfix({'Loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)

        #Calculate epoch metrics
        avg_loss = running_loss / (train_loader)
        metrics = self.matrics_calculator.calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions), 
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        return metrics


    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, learning_rate: float = 0.001,
              weight_decay: float = 1e-4, class_weights: Optional[torch.Tensor] = None,
              use_scheduler: bool = True, patience: int = 10) -> Dict:
        """Complete training loop"""

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))       
        else:
            criterion = nn.CrossEntropyLoss()
        
        #Setup scheduler
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=7, verbose=True
            )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=patience, mode='min')
        
        best_val_loss = float('inf')
        best_model_state = None
        
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)

            # Validation phase
            val_metrics = self.validate_epoch(val_loader, criterion, epoch)

            # Learning rate scheduling
            if use_scheduler:
                scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(epoch, train_metrics, val_metrics, is_best=True)

            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Store history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # Early stopping check
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Loaded best model weights")

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")

        # Close tensorboard writer
        self.writer.close()
        
        return self.history
    

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Comprehensive evaluation on test set"""
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc='Testing')
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)

                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().detach().numpy())
            
        # Calculate comprehensive metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            np.array(all_labels), 
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        
        # Generate and save confusion matrix
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        # Generate classification report
        self._generate_classification_report(test_metrics)
        
        return test_metrics
        

    def plot_confusion_matrix(self, y_true: List, y_pred: List, save_path: str = None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.metrics_calculator.class_names,
                   yticklabels=self.metrics_calculator.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(self, metrics: List[str] = ['loss', 'accuracy'], save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            train_key = f'train_{metric}'
            val_key = f'val_{metric}'
            
            if train_key in self.history and val_key in self.history:
                axes[i].plot(self.history[train_key], label=f'Train {metric}')
                axes[i].plot(self.history[val_key], label=f'Val {metric}')
                axes[i].set_title(f'{metric.capitalize()} History')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_history.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()