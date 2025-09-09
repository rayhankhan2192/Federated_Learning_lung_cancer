# Federated server

import argparse
from typing import Dict, List, Tuple, Optional, Union
import flwr as fl
import numpy as np
import torch
from datetime import datetime
import matplotlib
import logging
import sys
import os
import json
import time
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

class MedicalFLStrategy(fl.server.strategy.FedAvg):
    """
    FedAvg strategy extended with:
      - history tracking
      - best/last global checkpoint saving
      - detailed round logging & plots
    """
    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[fl.common.Parameters] = None,
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model_name = model_name
        self.num_classes = num_classes

        self.history = {
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
            "num_clients": [],
            "client_data_sizes": [],
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

        logger.info("FL Strategy initialized")
        logger.info(f"   → results_dir: {self.results_dir}")
        logger.info(f"   → model={self.model_name}, num_classes={self.num_classes}")

    def _save_strategy_config(self):
        config = {
            "strategy": "MedicalFLStrategy",
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "fraction_fit": self.fraction_fit,
            "fraction_evaluate": self.fraction_evaluate,
            "min_fit_clients": self.min_fit_clients,
            "min_evaluate_clients": self.min_evaluate_clients,
            "min_available_clients": self.min_available_clients,
            "accept_failures": self.accept_failures,
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(self.results_dir, "strategy_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def configure_fit(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        logger.info(f"Round {server_round}: configuring clients for training...")
        config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn else {}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        self.connected_clients.update({c.cid for c in clients})

        logger.info(f"Selected {len(clients)} clients: {sorted([c.cid for c in clients])}")
        fit_ins = fl.common.FitIns(parameters, config)
        return [(c, fit_ins) for c in clients]
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: fl.common.Parameters,
        client_manager: fl.server.client_manager.ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        if self.fraction_evaluate == 0.0:
            return []
        logger.info(f"Round {server_round}: configuring clients for evaluation...")
        config = self.on_evaluate_config_fn(server_round) if self.on_evaluate_config_fn else {}
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        logger.info(f"Selected {len(clients)} clients for evaluation")

        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(c, eval_ins) for c in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],

    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        t0 = time.time()
        logger.info(f"Round {server_round}: aggregating fit results "
                    f"(success={len(results)}, failures={len(failures)})")
        
        if len(results) < self.min_fit_clients:
            logger.warning(
                f"Not enough results to aggregate. "
                f"Expected {self.min_fit_clients}, got {len(results)}"
            )
            return None, {}
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, aggregated_metrics
        
        # Hold onto last aggregated params
        self.last_parameters = aggregated_parameters

        # Summaries from client metrics
        summary = self._calculate_fit_metrics(results)
        self.history["round"].append(server_round)
        self.history["train_loss"].append(summary["train_loss_avg"])
        self.history["train_accuracy"].append(summary["train_accuracy_avg"])
        self.history["train_f1"].append(summary["train_f1_avg"])
        self.history["val_loss"].append(summary["val_loss_avg"])
        self.history["val_accuracy"].append(summary["val_accuracy_avg"])
        self.history["val_f1"].append(summary["val_f1_avg"])
        self.history["num_clients"].append(len(results))
        self.history["client_data_sizes"].append(summary["client_data_sizes"])
        self.history["aggregation_time"].append(time.time() - t0)

        # Track best by validation F1
        if summary["val_f1_avg"] > self.best_f1:
            self.best_f1 = summary["val_f1_avg"]
            self.best_accuracy = summary["val_accuracy_avg"]
            self.best_round = server_round
            self.best_parameters = aggregated_parameters
            self.save_best_model()
            logger.info(f"🏆 New best model: round={self.best_round}, "
                        f"val_f1={self.best_f1:.4f}, val_acc={self.best_accuracy:.4f}")

        aggregated_metrics.update(summary)
        aggregated_metrics["aggregation_time"] = self.history["aggregation_time"][-1]
        self._log_round_summary(server_round, summary, len(results))

        # Periodic snapshot
        if server_round % 10 == 0:
            self.save_intermediate_results(server_round)
        return aggregated_parameters, aggregated_metrics
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
        logger.info(f"📊 Round {server_round}: aggregating evaluation results "
                    f"(success={len(results)} fail={len(failures)})")
        if not results:
            return None, {}
        test = self._calculate_eval_metrics(results)
        if len(self.history["test_loss"]) < len(self.history["round"]):
            self.history["test_loss"].append(test["test_loss_avg"])
            self.history["test_accuracy"].append(test["test_accuracy_avg"])
            self.history["test_f1"].append(test["test_f1_avg"])
        logger.info(f"   Test: loss={test['test_loss_avg']:.4f} "
                    f"acc={test['test_accuracy_avg']:.4f} f1={test['test_f1_avg']:.4f}")
        return test["test_loss_avg"], test