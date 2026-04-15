"""A collection of functions to apply the explainability method Grad-CAM."""

import torch
from captum.attr import LayerGradCam


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model(x)[0]           # [B, 768]
        logits = self.model.head(features)  # [B, 2]
        return logits


def list_conv_layers(model:torch.nn.Module) -> None:
    """Lists all convolutional layers in the model, as GradCAM is typically applied to a convolutional layer."""
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
            print(name, module)


def compute_gradcam_attribution(model:torch.nn.Module, input_sample:torch.Tensor, target_class:int, device:str="cpu"):
    """Compute GradCAM attribution for a given input sample and target class."""
    input_sample = input_sample.detach().to(device)
    input_sample.requires_grad_(True)

    wrapped_model = Wrapper(model).to(device).eval()

    # Check what layers are available for GradCAM
    #list_conv_layers(wrapped_model)   # There is only one conv layer

    target_layer = wrapped_model.model.patch_embed.proj  # The Conv2d layer in the patch embedding
    gradcam = LayerGradCam(wrapped_model, target_layer)

    # Get raw attributions (saliency map)
    raw_attr = gradcam.attribute(input_sample, target=target_class)
        
    return raw_attr