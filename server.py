# Federated server

import argparse
from typing import Dict, List, Tuple, Optional
import flwr as fl
import numpy as np
import torch
from datetime import datetime
import matplotlib
import logging
import sys
import os
from models.model_factory import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FL-Server")

def get_init_parameters(model_name: str, num_classes: int) -> fl.common.Parameters:
    """Get initial model parameters for FL server"""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        if models_dir not in sys.path:
            sys.path.insert(0, models_dir)
        model = get_model(model_name, num_classes, pretrained=False)
        parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

        logger.info(f"Initial model parameters loaded for : {model_name}")
        logger.info(f"Server Model parameters shapes:")
        total_params = 0
        for name, param in model.state_dict().items():
            param_size = param.numel()
            logger.info(f" {name}: shape={param.shape}, size: {param_size}")
            total_params += param_size
        logger.info(f"Total number of parameters: {total_params}")
        return fl.common.ndarrays_to_parameters(parameters)
    except Exception as e:
        logger.error(f"❌ Failed to get initial parameters: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def fit_config(server_round: int)->Dict[str, fl.common.Scalar]:
    """Per-round training config broadcast to clients."""
    config = {
        "serever_round": server_round,
        "local_epochs": 5,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss_function": "cross_entropy",
        "optimizer": "adamw",
        "scheduler": "plateau",
        "use_scheduler": True,
        "batch_size": 32,
    }
    if server_round > 20:
        config["loss_function"] = "focal"
    if 32 <= server_round <= 60:
        config["learning_rate"] = 5e-4
        config["local_epochs"] = 4
    elif 61 <= server_round <= 80:
        config["learning_rate"] = 2e-4
        config["local_epochs"] = 3
    elif server_round > 80:
        config["learning_rate"] = 1e-4
        config["local_epochs"] = 2
    
    logger.info(
        f"Round {server_round} training config: "
        f"epochs: {config['local_epochs']}, lr: {config['learning_rate']}, loss: {config['loss_function']}"
    )
    return config


def evaluate_config(server_round: int)->Dict[str, fl.common.Scalar]:
    return {"server_round": server_round}

def wighted_average(metrics: List[Tuple[int, Dict]])->Dict:
    """Weighted average across client metrics dictionaries."""
    logger.info(f"Aggregating metrics from {len(metrics)} clients")
    if not metrics:
        return {}
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
    aggregated_metrics: Dict[str, float] = {}
    for num_samples, client_metrics in metrics:
        weight = num_samples / total_samples
        for key, value in client_metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + weight * float(value)
    return aggregated_metrics

class FedaratedStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy extended with:
      - history tracking
      - best/last global checkpoint saving
      - detailed round logging & plots
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,

    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            evaluate_metrics_aggregation_fn=wighted_average,
        )
        self.model_name = model_name
        self.num_classes = num_classes
        self.history: Dict[str, List] = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "num_Clients": [],
            "client_data_size": [],
            "aggregation_time": [],
        }
        self.best_accuracy = 0.0
        self.best_f1 = 0.0
        self.best_round = 0
        self.best_parameters: Optional[fl.common.Parameters] = None
        self.last_parameters: Optional[fl.common.Parameters] = None

        self.connected_clients = set()
        self.client_metrics_history = {}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"fl_results_{ts}"
        os.makedirs(self.results_dir, exist_ok=True)
        self._save_strategy_config()

        logger.info("✅ FL Strategy initialized")
        logger.info(f"   → results_dir: {self.results_dir}")
        logger.info(f"   → model={self.model_name}, num_classes={self.num_classes}")

