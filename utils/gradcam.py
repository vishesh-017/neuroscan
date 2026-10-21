"""
Explainable AI (XAI): Grad-CAM Implementation
==============================================
Gradient-weighted Class Activation Mapping for PyTorch CNNs.
Visualizes which anatomical regions in the MRI scan drove the model's decision.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM implementation for interpreting convolutional neural network decisions.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register PyTorch forward & backward hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        """
        Generates normalized Grad-CAM heatmap for a given input image tensor.
        
        Args:
            input_tensor (torch.Tensor): Shape (1, C, H, W)
            target_class (int, optional): Target class index. If None, uses top prediction.
            
        Returns:
            heatmap (np.ndarray): 2D float array in [0, 1] of shape (H, W)
        """
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Zero gradients
        self.model.zero_grad()
        # Backward pass on target class score
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        # Global Average Pooling of gradients to calculate channel weights alpha_k
        # gradients shape: (1, channels, h, w)
        grads = self.gradients.detach().cpu().numpy()[0]
        acts = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(grads, axis=(1, 2))  # (channels,)
        
        # Weighted sum of feature maps
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
            
        # Apply ReLU: only positive contributions matter for the selected class
        cam = np.maximum(cam, 0)
        
        # Resize to input tensor size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (w, h))
        
        # Normalize between 0 and 1
        denom = (cam.max() - cam.min()) + 1e-8
        cam = (cam - cam.min()) / denom
        return cam, target_class


def overlay_gradcam(original_rgb_np, heatmap, alpha=0.45):
    """
    Overlays a Grad-CAM heatmap onto the original RGB image using JET colormap.
    
    Args:
        original_rgb_np (np.ndarray): RGB image in [0, 1] or [0, 255]
        heatmap (np.ndarray): 2D float heatmap in [0, 1]
        alpha (float): Opacity of heatmap (0.0 to 1.0)
        
    Returns:
        overlaid_rgb (np.ndarray): RGB image in uint8 [0, 255]
    """
    if original_rgb_np.max() <= 1.0:
        base_img = (original_rgb_np * 255).astype(np.uint8)
    else:
        base_img = original_rgb_np.astype(np.uint8)
        
    # Apply JET colormap: red = high focus, blue = low focus
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize colormap if needed
    if colored_heatmap.shape[:2] != base_img.shape[:2]:
        colored_heatmap = cv2.resize(colored_heatmap, (base_img.shape[1], base_img.shape[0]))
        
    # Blend images
    blended = cv2.addWeighted(base_img, 1.0 - alpha, colored_heatmap, alpha, 0)
    return blended