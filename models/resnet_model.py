import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

class MedicalResNet50(nn.Module):
    """
    ResNet50 model for more complex medical image analysis
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5):
        super(MedicalResNet50, self).__init__()
        
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=3, bias=False)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )

        self._initialize_weights()
