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
        Train model locally using federated learning round configuration.

        Args:
            parameters: Global model parameters from server
            config: Training configuration from server

        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Client {self.client_id}: Starting local training round")

        #Set global weights
        self.set_parameters(parameters)

        #Config extraction
        local_epochs = config.get("local_epochs", self.local_epochs)
        learning_rate = config.get("learning_rate", self.learning_rate)
        weight_decay = config.get("weight_decay", 1e-4)
        loss_function = config.get("loss_function", "crossentropy")
        optimizer_name = config.get("optimizer", "adamw")
        scheduler_name = config.get("scheduler", "plateau")
        use_scheduler = config.get("use_scheduler", True)

        #Loss Function
        if loss_function == "focal":
            criterion = FocalLoss(alpha=1.0, gamma=2.0)
        elif loss_function == "label_smoothing":
            criterion = LabelSmoothingLoss(num_classes=self.num_classes, smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))

        # Call ModelTrainer.train
        train_history = self.trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            num_epochs=local_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            class_weights=self.class_weights,
            use_scheduler=use_scheduler,
            patience=10,
            criterion=criterion,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name
        )

        # === Evaluation after training
        test_metrics = self.trainer.evaluate(self.test_loader)

        # === Save model (optional)
        model_path = f"client_{self.client_id}_best_model.pth"
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Client {self.client_id}: Best model saved to {model_path}")

        # === Final Metrics
        metrics = {
            "train_loss": train_history["train_loss"][-1],
            "train_accuracy": train_history["train_accuracy"][-1],
            "val_loss": train_history["val_loss"][-1],
            "val_accuracy": train_history["val_accuracy"][-1],
            "val_f1": train_history["val_f1_macro"][-1],
            "test_accuracy": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
            "num_examples": len(self.train_loader.dataset)
        }

        logger.info(f"Client {self.client_id}: Local training completed")
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