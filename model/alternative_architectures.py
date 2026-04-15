""" 
Alternative, lightweight architectures for video classification.

There may be other architectures than EndoFM-LV (121M parameters)
that are better suited for small endoscopic video datasets such as PolypGen.

Architectures included:
1. X3D-M: Efficient 3D CNN

You may want to add your own architecture.
"""

import torch
from pytorchvideo.models.x3d import create_x3d
import logging


# =============================================================
# X3D-M
# =============================================================
class X3D_M(torch.nn.Module):
    """The X3D architecture with X3D-M scaling, modified to fit PolypGen data 
        - input_clip_length of 6 instead of 16
        - model_num_class of 2 (binary classification) instead of 400

        A lightweight 3D CNN for video classification.

        Paper: "X3D: Expanding Architectures for Efficient Video Recognition" (CVPR 2020)
    """

    def __init__(self):
        super().__init__()
        # the backbone (ResNet) with binary classification head
        self.model = create_x3d(
            input_clip_length=6,     # X3D-M originally uses 16
            input_crop_size=224,
            input_channel=3,
            model_num_class=2,
            dropout_rate=0.5,
            width_factor=2.0,
            depth_factor=2.2,        # X3D-M specific scaling
            bottleneck_factor=2.25,
            se_ratio=0.0625,
            head_dim_out=2048,
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Forward pass of an input tensor 
        of shape (B, C, T, H, W), i.e. (B, 3, 6, 224, 224).
        """
        output = self.model(x)
        return output


# =============================================================
# Model Factory
# =============================================================
def get_architecture(name:str) -> torch.nn.Module:
    """Factory function to create alternative architectures."""
    # Available architectures
    architectures = {
        'x3d': X3D_M,
    }

    if name.lower() not in architectures:
        available = ', '.join(architectures.keys())
        raise ValueError(f"Unknown architecture: {name}. Available: {available}.")

    # Get architecture
    model = architectures[name.lower()]()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Created {name} architecture:")
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    return model