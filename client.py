import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict
import argparse
import logging
from typing import Dict, List, Tuple
import os
import warnings
warnings.filterwarnings("ignore")

# Import custom modules
from models.resnet_model import get_model, FocalLoss, LabelSmoothingLoss
from utils.dataloder import create_data_loaders, get_class_weights
from utils.train_eval import ModelTrainer, ModelMetrics, get_optimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical image classification
    """
    
    def __init__(self, client_id: int, data_dir: str, device: torch.device,
                 model_name: str = "resnet50", num_classes: int = 3,
                 batch_size: int = 32, local_epochs: int = 8):
        """
        Initialize FL client
        
        Args:
            client_id: Unique identifier for this client
            data_dir: Path to client's local data
            device: Computing device (CPU/GPU)
            model_name: Model architecture to use
            num_classes: Number of classification classes
            batch_size: Batch size for training
            local_epochs: Number of local training epochs per round
        """
        self.client_id = client_id
        self.data_dir = data_dir
        self.device = device
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Initialize model
        self.model = get_model(model_name, num_classes, pretrained=True)
        self.model.to(device)
        
        # Create data loaders
        logger.info(f"Client {client_id}: Loading data from {data_dir}")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir=data_dir,
            batch_size=batch_size,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
            image_size=(224, 224),
            num_workers=1
        )
        
        # Calculate class weights for handling imbalanced data
        self.class_weights = get_class_weights(self.train_loader)
        logger.info(f"Client {client_id}: Class weights: {self.class_weights}")
        
        # Initialize trainer
        save_dir = f"client_{client_id}_checkpoints"
        log_dir = f"client_{client_id}_logs"
        self.trainer = ModelTrainer(self.model, device, save_dir, log_dir)
        
        # Training configuration
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        
        logger.info(f"Client {client_id} initialized successfully")
        logger.info(f"  - Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"  - Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"  - Test samples: {len(self.test_loader.dataset)}")
    
    def get_parameters(self, config: Dict = None) -> List[np.ndarray]:
        """
        Get model parameters as numpy arrays
        
        Args:
            config: Configuration dictionary from server
            
        Returns:
            List of model parameters as numpy arrays
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays
        
        Args:
            parameters: List of model parameters as numpy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    