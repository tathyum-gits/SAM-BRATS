import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

def dice_loss(pred, target):
    """Calculate Dice loss."""
    smooth = 1e-6
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def boundary_loss(pred, target):
    """
    Calculate boundary loss using distance transform.
    """
    pred = torch.sigmoid(pred)
    target = target > 0.5
    
    # Compute distance transform of the target mask
    distance_map = torch.tensor(
        distance_transform_edt(target.cpu().numpy())
    ).to(target.device)
    
    # Compute the boundary loss
    loss = (distance_map * (1 - pred) ** 2).mean()
    return loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Sigmoid activation
        probs = torch.sigmoid(logits)
        
        # Cross-entropy term
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        # Compute Focal Loss
        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    """Combined loss using Dice, Boundary, and Focal losses."""
    
    def __init__(self, config):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(
            alpha=config['loss']['focal_alpha'],
            gamma=config['loss']['focal_gamma']
        )
        self.dice_weight = config['loss']['dice_weight']
        self.boundary_weight = config['loss']['boundary_weight']
        self.focal_weight = config['loss']['focal_weight']

    def forward(self, logits, targets):
        dice = dice_loss(logits, targets)
        boundary = boundary_loss(logits, targets)
        focal = self.focal_loss(logits, targets)

        # Weighted sum of losses
        loss = (
            self.dice_weight * dice +
            self.boundary_weight * boundary +
            self.focal_weight * focal
        )
        return loss

def dice_score(pred, target, threshold=0.2):
    """Calculate Dice score for evaluation."""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2.0 * intersection) / (pred.sum() + target.sum() + 1e-6)

def precision(pred, target, threshold=0.2):
    """Calculate precision score."""
    pred = torch.sigmoid(pred) > threshold
    preds_flat = pred.view(-1).cpu().numpy()
    targets_flat = target.view(-1).cpu().numpy()
    return precision_score(targets_flat, preds_flat, zero_division=0)

def mae(pred, target):
    """Calculate Mean Absolute Error."""
    pred = torch.sigmoid(pred)
    return torch.abs(pred - target).mean().item()
