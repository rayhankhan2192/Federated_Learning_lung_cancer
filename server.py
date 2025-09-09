# Federated server

import argparse
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
        logger.info(f"Round {server_round}: aggregating evaluation results "
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
    

    def _calculate_fit_metrics(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]]) -> Dict:
        """Calculate weighted average training/validation metrics from all clients."""

        total_examples = sum(fit_res.num_examples for _, fit_res in results) or 1
        metric_keys = ["train_loss", "train_accuracy", "train_f1",
                    "val_loss", "val_accuracy", "val_f1"]

        weighted_sums = {key: 0.0 for key in metric_keys}
        client_data_sizes = []
        client_metrics_list = []

        for client_proxy, fit_res in results:
            weight = fit_res.num_examples / total_examples
            client_data_sizes.append(fit_res.num_examples)

            metrics_for_client = {}
            for key in metric_keys:
                value = fit_res.metrics.get(key, 0.0) if fit_res.metrics else 0.0
                weighted_sums[key] += float(value) * weight
                metrics_for_client[key] = float(value)

            client_metrics_list.append({
                "client_id": client_proxy.cid,
                "num_examples": fit_res.num_examples,
                "metrics": metrics_for_client,
            })

        # Save per-round client metrics
        current_round = len(self.history["round"]) + 1
        self.client_metrics_history[current_round] = client_metrics_list

        return {
            "train_loss_avg": weighted_sums["train_loss"],
            "train_accuracy_avg": weighted_sums["train_accuracy"],
            "train_f1_avg": weighted_sums["train_f1"],
            "val_loss_avg": weighted_sums["val_loss"],
            "val_accuracy_avg": weighted_sums["val_accuracy"],
            "val_f1_avg": weighted_sums["val_f1"],
            "total_examples": total_examples,
            "client_data_sizes": client_data_sizes,
            "num_participating_clients": len(results),
        }
    
    def _calculate_eval_metrics(
        self, results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]]
    ) -> Dict:
        """Calculate weighted average evaluation metrics from all clients."""

        total_examples = sum(eval_res.num_examples for _, eval_res in results) or 1

        weighted_loss = 0.0
        weighted_accuracy = 0.0
        weighted_f1 = 0.0

        for _, eval_res in results:
            weight = eval_res.num_examples / total_examples
            weighted_loss += float(eval_res.loss or 0.0) * weight

            if eval_res.metrics:
                weighted_accuracy += float(eval_res.metrics.get("accuracy", 0.0)) * weight
                weighted_f1 += float(eval_res.metrics.get("f1_macro", 0.0)) * weight

        return {
            "test_loss_avg": weighted_loss,
            "test_accuracy_avg": weighted_accuracy,
            "test_f1_avg": weighted_f1,
            "total_test_examples": total_examples,
            "num_eval_clients": len(results),
        }
    
    def _log_round_summary(self, round_num: int, summary: Dict, num_clients: int):
        logger.info("=" * 80)
        logger.info(f"ROUND {round_num} SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Clients: {num_clients} | examples: {summary['total_examples']:,}")
        logger.info(f"Train: loss={summary['train_loss_avg']:.4f} acc={summary['train_accuracy_avg']:.4f} f1={summary['train_f1_avg']:.4f}")
        logger.info(f"Val  : loss={summary['val_loss_avg']:.4f} acc={summary['val_accuracy_avg']:.4f} f1={summary['val_f1_avg']:.4f}")
        logger.info(f"Best : round={self.best_round} val_f1={self.best_f1:.4f} val_acc={self.best_accuracy:.4f}")
        logger.info("=" * 80)

    def save_best_model(self) -> None:
        """Save the best global model checkpoint based on validation F1 score."""
        if self.best_parameters is None:
            logger.warning("No best parameters available, skipping model save.")
            return

        try:
            # Ensure models folder is on sys.path for dynamic import
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            # Rebuild the model and load the best parameters
            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)
            param_pairs = zip(
                model.state_dict().keys(),
                fl.common.parameters_to_ndarrays(self.best_parameters),
            )
            best_state_dict = OrderedDict({name: torch.tensor(arr) for name, arr in param_pairs})

            # Build checkpoint dictionary
            checkpoint = {
                "round": self.best_round,
                "model_state_dict": best_state_dict,
                "best_f1": self.best_f1,
                "best_accuracy": self.best_accuracy,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            # Save to file
            save_path = os.path.join(self.results_dir, f"best_model_round_{self.best_round}.pth")
            torch.save(checkpoint, save_path)

            logger.info(f"Best model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save best model: {exc}")


    def save_last_model(self) -> None:
        """Save the most recent global model checkpoint (last round)."""
        if self.last_parameters is None:
            logger.warning("No last parameters available, skipping model save.")
            return

        try:
            # Ensure models folder is on sys.path for dynamic import
            models_dir = os.path.join(os.path.dirname(__file__), "models")
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            # Build model and load last parameters
            model = get_model(self.model_name, num_classes=self.num_classes, pretrained=False)
            param_pairs = zip(
                model.state_dict().keys(),
                fl.common.parameters_to_ndarrays(self.last_parameters),
            )
            last_state_dict = OrderedDict({name: torch.tensor(arr) for name, arr in param_pairs})

            # Build checkpoint
            checkpoint = {
                "model_state_dict": last_state_dict,
                "model_name": self.model_name,
                "num_classes": self.num_classes,
                "timestamp": datetime.now().isoformat(),
            }

            # Save to file
            save_path = os.path.join(self.results_dir, "last_global_model.pth")
            torch.save(checkpoint, save_path)

            logger.info(f"Last global model saved successfully → {save_path}")

        except Exception as exc:
            logger.error(f"Failed to save last global model: {exc}")

    def save_intermediate_results(self, rnd: int):
        try:
            with open(os.path.join(self.results_dir, f"history_round_{rnd}.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_dir, f"client_metrics_round_{rnd}.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix=f"_round_{rnd}")
            logger.info(f"Intermediate results saved for round {rnd}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")

    def save_final_results(self):
        try:
            with open(os.path.join(self.results_dir, "final_training_history.json"), "w") as f:
                json.dump(self.history, f, indent=2)
            with open(os.path.join(self.results_dir, "final_client_metrics.json"), "w") as f:
                json.dump(self.client_metrics_history, f, indent=2)
            self.plot_training_curves(save_suffix="_final")
            self._generate_final_report()
            logger.info(f"Final results saved in {self.results_dir}")
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")

    def plot_training_curves(self, save_suffix: str = ""):
        if not self.history["round"]:
            return
        rounds = self.history["round"]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Loss
        axes[0, 0].plot(rounds, self.history["train_loss"], label="Train Loss", linewidth=2, marker="o")
        axes[0, 0].plot(rounds, self.history["val_loss"], label="Val Loss", linewidth=2, marker="s")
        if self.history["test_loss"]:
            axes[0, 0].plot(rounds, self.history["test_loss"], label="Test Loss", linewidth=2, marker="^")
        axes[0, 0].set_title("Loss")

        # Acc
        axes[0, 1].plot(rounds, self.history["train_accuracy"], label="Train Acc", linewidth=2, marker="o")
        axes[0, 1].plot(rounds, self.history["val_accuracy"], label="Val Acc", linewidth=2, marker="s")
        if self.history["test_accuracy"]:
            axes[0, 1].plot(rounds, self.history["test_accuracy"], label="Test Acc", linewidth=2, marker="^")
        axes[0, 1].set_title("Accuracy")

        # F1
        axes[0, 2].plot(rounds, self.history["train_f1"], label="Train F1", linewidth=2, marker="o")
        axes[0, 2].plot(rounds, self.history["val_f1"], label="Val F1", linewidth=2, marker="s")
        if self.history["test_f1"]:
            axes[0, 2].plot(rounds, self.history["test_f1"], label="Test F1", linewidth=2, marker="^")
        axes[0, 2].set_title("F1-Score")

        # Clients per round
        axes[1, 0].bar(rounds, self.history["num_clients"], alpha=0.8)
        axes[1, 0].set_title("Clients per Round")

        # Aggregation time
        if self.history["aggregation_time"]:
            axes[1, 1].plot(rounds, self.history["aggregation_time"], linewidth=2, marker="d")
            axes[1, 1].set_title("Aggregation Time (s)")

        # Data distribution (latest round)
        if self.history["client_data_sizes"]:
            latest = self.history["client_data_sizes"][-1]
            labels = [f"C{i+1}" for i in range(len(latest))]
            axes[1, 2].pie(latest, labels=labels, autopct="%1.1f%%", startangle=90)
            axes[1, 2].set_title("Data Distribution (latest)")

        for ax in axes.ravel():
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
        plt.tight_layout()
        out = os.path.join(self.results_dir, f"training_curves{save_suffix}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 Saved plot → {out}")