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
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
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
                 model_name: str = "customcnn", num_classes: int = 3,
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
            num_workers=3
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
    
    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model locally using federated learning round configuration
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
            
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training round")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Get training configuration from server
        local_epochs = config.get("local_epochs", self.local_epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)
        loss_function = config.get("loss_function", "crossentropy")
        
        # Setup optimizer
        optimizer = get_optimizer(self.model, "adam", learning_rate, self.weight_decay)
        
        # Setup loss function
        if loss_function == "focal":
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_function == "label_smoothing":
            criterion = LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        
        # Local training
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                epoch_samples += data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                epoch_correct += (predicted == target).sum().item()
            
            epoch_accuracy = epoch_correct / epoch_samples
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{local_epochs}: "
                       f"Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
        
        # Calculate final metrics
        avg_loss = total_loss / (local_epochs * len(self.train_loader))
        accuracy = correct_predictions / total_samples
        
        # Validate on local validation set
        val_metrics = self._evaluate_local()
        
        # Prepare metrics to send to server
        metrics = {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_macro"],
            "num_examples": len(self.train_loader.dataset)
        }
        
        logger.info(f"Client {self.client_id}: Local training completed")
        logger.info(f"  - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.4f}")
        logger.info(f"  - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        return self.get_parameters(), len(self.train_loader.dataset), metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test set
        
        Args:
            parameters: Global model parameters from server
            config: Evaluation configuration from server
            
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting evaluation")
        
        # Set global parameters
        self.set_parameters(parameters)
        
        # Evaluate on test set
        test_metrics = self.trainer.evaluate(self.test_loader)
        
        logger.info(f"Client {self.client_id}: Evaluation completed")
        logger.info(f"  - Test Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"  - Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
        
        return (test_metrics.get("loss", 0.0), 
                len(self.test_loader.dataset), 
                test_metrics)
    
    def _evaluate_local(self) -> Dict:
        """
        Evaluate model on local validation set using ModelMetrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        criterion = nn.CrossEntropyLoss()
        metrics_calculator = ModelMetrics()
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                total_loss += loss.item()
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().detach().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate comprehensive metrics using ModelMetrics
        metrics = metrics_calculator.calculate_metrics(
            np.array(all_labels),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
        metrics['loss'] = avg_loss
        
        return metrics

def create_client(client_id: int, data_dir: str, model_name: str = "customcnn") -> MedicalFLClient:
    """
    Factory function to create FL client
    
    Args:
        client_id: Unique client identifier
        data_dir: Path to client's data directory
        model_name: Model architecture name
        
    Returns:
        Initialized MedicalFLClient
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create client
    client = MedicalFLClient(
        client_id=client_id,
        data_dir=data_dir,
        device=device,
        model_name=model_name,
        num_classes=3,
        batch_size=32,
        local_epochs=50
    )
    
    return client

# def main():
#     """Main function to run FL client"""
#     parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Imaging")
#     parser.add_argument("--client-id", type=int, default=1, help="Client ID")
#     parser.add_argument("--data-dir", type=str, required=True, help="Path to client data directory")
#     parser.add_argument("--server-address", type=str, default="localhost:8080", help="FL server address")
#     parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="Model architecture")
    
#     args = parser.parse_args()
    
#     # Validate data directory
#     if not os.path.exists(args.data_dir):
#         raise ValueError(f"Data directory not found: {args.data_dir}")
    
#     # Create client
#     client = create_client(args.client_id, args.data_dir, args.model)
    
#     # Start FL client
#     logger.info(f"Starting FL client {args.client_id} connecting to {args.server_address}")
#     fl.client.start_numpy_client(
#         server_address=args.server_address,
#         client=client
#     )

def main():
    """Main function to run FL client"""
    parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Imaging")
    parser.add_argument("--client-id", type=int, default=1, help="Client ID")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to client data directory")
    parser.add_argument("--server-address", type=str, default="localhost:8080", help="FL server address")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet50", "customcnn"], help="Model architecture")
    parser.add_argument("--train-local", action="store_true", help="Run local training only (no FL server)")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory not found: {args.data_dir}")

    client = create_client(args.client_id, args.data_dir, args.model)

    if args.train_local:
        #Local training only
        logger.info("Running standalone local training (no FL server)")
        # client.fit(client.get_parameters(), config={})

        updated_params, num_examples, train_metrics = client.fit(client.get_parameters(), config={})
        
        # Run evaluation on test set
        test_loss, test_examples, test_metrics = client.evaluate(updated_params, config={})
        
        logger.info("Local training and evaluation completed:")
        logger.info(f"  - Final train metrics: {train_metrics}")
        logger.info(f"  - Final test metrics: {test_metrics}")
        return

    #Federated client
    logger.info(f"Starting FL client {args.client_id} connecting to {args.server_address}")
    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client()
    )

if __name__ == "__main__":
    main()