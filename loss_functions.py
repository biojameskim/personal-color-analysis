import torch
import torch.nn.functional as F


def joint_loss(outputs, season_targets, subtype_targets, full_targets=None, alpha=0.4, beta=0.4, gamma=0.2):
    """
    Calculate joint loss with components for season prediction, subtype prediction, and full class prediction.

    Args:
        outputs (dict): Model outputs from forward pass
        season_targets (torch.Tensor): Season labels (0-3)
        subtype_targets (torch.Tensor): Subtype labels (0-2)
        full_targets (torch.Tensor, optional): Combined labels (0-11)
        alpha (float): Weight for season loss
        beta (float): Weight for subtype loss
        gamma (float): Weight for full classification loss

    Returns:
        torch.Tensor: Combined loss value
    """
    full_targets = season_targets * 3 + subtype_targets # We have 12 total classes (full targets)

    # Calculate individual losses
    season_loss = F.cross_entropy(outputs['season_logits'], season_targets)
    subtype_loss = F.cross_entropy(outputs['subtype_logits'], subtype_targets)
    full_loss = F.cross_entropy(outputs['full_logits'], full_targets)

    # Combine losses with weights
    total_loss = alpha * season_loss + beta * subtype_loss + gamma * full_loss

    return total_loss, season_loss, subtype_loss, full_loss