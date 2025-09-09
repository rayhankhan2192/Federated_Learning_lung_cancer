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
