 
"""
Grad-CAM for visualizing CNN decisions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2

def generate_gradcam(model, image, target_layer):
    """
    Generate Grad-CAM heatmap
    
    Args:
        model: trained PyTorch model
        image: input tensor (1, C, H, W)
        target_layer: layer to visualize (e.g., model.features[-1])
    
    Returns:
        heatmap: numpy array (H, W)
    """
    # Store gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    model.eval()
    output = model(image)
    pred_class = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()
    
    # Get gradients and activations
    grad = gradients[0].cpu().data.numpy()[0]
    act = activations[0].cpu().data.numpy()[0]
    
    # Compute weights
    weights = np.mean(grad, axis=(1, 2))
    
    # Weighted combination
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]
    
    # ReLU and normalize
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    return cam

def overlay_heatmap(image, heatmap, alpha=0.4):
    """
    Overlay heatmap on original image
    
    Args:
        image: original image (H, W, 3) - numpy array
        heatmap: Grad-CAM heatmap (H, W)
        alpha: transparency of heatmap
    
    Returns:
        overlaid image
    """
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Overlay
    overlaid = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlaid