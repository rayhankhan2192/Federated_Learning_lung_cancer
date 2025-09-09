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