import torch
import torch.nn as nn
from torchvision import models

class DenseNet121Medical(nn.Module):
    """
    DenseNet121 model for medical image classification using CheXpert/RadImageNet pretrained weights.
    Converts first convolution to accept 1-channel input by averaging pretrained weights.
    """
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout_rate: float = 0.5, dataset: str = "chexpert"):
        super(DenseNet121Medical, self).__init__()

        # Load pre-trained DenseNet121 (CheXpert or RadImageNet)
        if dataset == "chexpert":
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.CHEXPERT if pretrained else None)
        elif dataset == "radimagenet":
            # Assuming RadImageNet pretrained weights are available in torchvision or from an external source
            self.densenet = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            self.densenet = models.densenet121(pretrained=pretrained)
        
        # Modify the first convolution to accept 1-channel input
        with torch.no_grad():
            w = self.densenet.features.conv0.weight  # [64, 3, 7, 7]
            self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.densenet.features.conv0.weight.copy_(w.mean(dim=1, keepdim=True))  # Average RGB weights

        # Get the output features from DenseNet and define the final classifier
        self.densenet.classifier = nn.Identity()  # Use DenseNet as a feature extractor
        self.classifier = nn.Sequential(
            nn.Linear(self.densenet.classifier.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),  # Final classification layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DenseNet121 model.
        x: [B, 1, H, W] (grayscale) or [B, 3, H, W] (RGB)
        """
        # Convert grayscale to RGB if needed (DenseNet expects RGB input)
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        features = self.densenet(x_rgb)  # Extract features from DenseNet
        return self.classifier(features)  # Classify using the classifier head

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the DenseNet backbone without classification.
        """
        x_rgb = x if x.size(1) == 3 else x.repeat(1, 3, 1, 1)
        return self.densenet(x_rgb)  # Return features without classification
