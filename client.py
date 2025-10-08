import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" 
import logging, warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

import torch
import torch.nn as nn
import flwr as fl
import numpy as np
from collections import OrderedDict
import argparse
import logging
from typing import Dict, List, Tuple
import os
import cv2
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

# Import custom modules
from models.model_factory import get_model, FocalLoss, LabelSmoothingLoss
from utils.dataloder import create_data_loaders, get_class_weights
from utils.train_eval import ModelTrainer, ModelMetrics, get_optimizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_BASE_DIR = os.path.abspath("Result/ClientResults")
os.makedirs(RESULTS_BASE_DIR, exist_ok=True)


def _normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    a -= a.min(); a += 1e-12
    a /= a.max()
    return a

def _find_last_conv(module: nn.Module) -> nn.Conv2d:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last

class _GradCAM:
    """Minimal, fast Grad-CAM for 1-ch CT; works with CustomCNN/ResNet50."""
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.tl = target_layer
        self.A = None
        self.dA = None
        self.ha = self.tl.register_forward_hook(self._hook_act)
        self.hg = self.tl.register_full_backward_hook(self._hook_grad)

    def _hook_act(self, module, inp, out):
        self.A = out
    def _hook_grad(self, module, gin, gout):
        self.dA = gout[0]

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        A = self.A[0]                 # [C,H,W]
        dA = self.dA[0]               # [C,H,W]
        w = dA.mean(dim=(1,2))        # [C]
        cam = torch.relu((w[:,None,None] * A).sum(dim=0)).detach().cpu().numpy()
        return _normalize01(cam)

    def close(self):
        self.ha.remove(); self.hg.remove()

def _overlay_on_gray(img_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """img_u8: HxW uint8; heat: HxW [0..1]; returns HxWx3 BGR (OpenCV)"""
    H, W = img_u8.shape
    heat_r = cv2.resize(heat, (W, H))
    heatmap = cv2.applyColorMap((heat_r*255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 1.0, heatmap, alpha, 0)

def _deletion_curve_scores(model: nn.Module, x: torch.Tensor, heat: np.ndarray, steps: int = 10) -> List[float]:
    """Iteratively zero most-important pixels; record target logit."""
    device = next(model.parameters()).device
    x = x.clone().to(device)
    with torch.no_grad():
        base_logits = model(x)[0]
        cls = int(base_logits.argmax().item())
        scores = [base_logits[cls].item()]

    H, W = heat.shape
    order = np.argsort(-heat.flatten())  # descending
    k = int(np.ceil(len(order) / steps))
    for s in range(steps):
        idxs = order[s*k:(s+1)*k]
        for idx in idxs:
            y, z = idx // W, idx % W
            x[0, 0, y, z] = 0.0
        with torch.no_grad():
            scores.append(model(x)[0, cls].item())
    return scores

def _auc_trapz(y: List[float]) -> float:
    y = np.asarray(y, dtype=np.float32)
    x = np.linspace(0, 1, len(y), dtype=np.float32)
    return float(np.trapz(y, x))


class MedicalFLClient(fl.client.NumPyClient):
    """
    Federated Learning client for medical image classification
    """
    
    def __init__(
        self, 
        client_id: int, 
        data_dir: str, 
        device: torch.device,
        model_name: str = "customcnn", 
        num_classes: int = 3,
        batch_size: int = 32, 
        local_epochs: int = 8,
        results_base_dir: str = RESULTS_BASE_DIR,
        ):
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
        self.results_dir = results_base_dir
        
        # Initialize model
        self.model = get_model(model_name, num_classes, pretrained=True)
        self.model.to(device)
        os.makedirs(self.xai_dir, exist_ok=True)

        # XAI: pick last conv layer once
        self.target_layer = _find_last_conv(self.model)

        self.xai_dir = os.path.join(self.results_dir, f"client_{client_id}_xai")


        
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
        #self.results_dir = os.path.join(results_base_dir, f"client_{client_id}_checkpoints")
        save_dir = os.path.join(results_base_dir, f"client_{client_id}_checkpoints")
        log_dir = os.path.join(results_base_dir, f"client_{client_id}_logs")
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

        xai_metrics = self._xai_probe(self.val_loader, num_samples=16, save_k=3)

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
            "num_examples": len(self.train_loader.dataset),
            **xai_metrics, 
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
        # Optional: run a quick XAI probe during evaluate as well
        xai_metrics = self._xai_probe(self.val_loader, num_samples=12, save_k=0)
        test_metrics.update(xai_metrics)
        
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
    
    def _xai_probe(self, loader, num_samples: int = 16, save_k: int = 3) -> Dict:
        """
        Runs Grad-CAM on a small subset, computes Deletion-AUC faithfulness,
        and (optionally) saves a few overlays locally. Returns numeric metrics only.
        """
        self.model.eval()
        device = self.device
        cam_engine = _GradCAM(self.model, self.target_layer)

        del_aucs = []
        saved = 0
        seen = 0

        with torch.no_grad():
            for data, target in loader:
                # data: [B,1,224,224], target: [B]
                for i in range(data.size(0)):
                    x = data[i:i+1].to(device)  # [1,1,H,W]

                    # forward predict class
                    logits = self.model(x)
                    pred_idx = int(torch.argmax(logits, dim=1).item())

                    # Grad-CAM heat
                    heat = cam_engine.generate(x, class_idx=pred_idx)  # [H,W] float [0..1]

                    # Faithfulness: deletion curve AUC (lower is better)
                    scores = _deletion_curve_scores(self.model, x, heat, steps=10)
                    del_aucs.append(_auc_trapz(scores))

                    # Optionally save a few overlays (local only)
                    if saved < save_k:
                        # input tensor back to uint8 for visualization
                        # assuming your dataloader already returns normalized [0,1] or [-, +]?
                        # safest: rescale from tensor directly
                        img = data[i, 0].cpu().numpy()
                        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
                        img_u8 = (img * 255).astype(np.uint8)

                        overlay_bgr = _overlay_on_gray(img_u8, heat, alpha=0.35)
                        out_path = os.path.join(self.xai_dir, f"round_overlay_{saved+1}.png")
                        cv2.imwrite(out_path, overlay_bgr)
                        saved += 1

                    seen += 1
                    if seen >= num_samples:
                        break
                if seen >= num_samples:
                    break

        cam_engine.close()

        # Aggregate metrics
        if len(del_aucs) == 0:
            return {"xai_del_auc_mean": 0.0, "xai_del_auc_std": 0.0}
        return {
            "xai_del_auc_mean": float(np.mean(del_aucs)),
            "xai_del_auc_std": float(np.std(del_aucs)),
        }


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