import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class HierarchicalSoftmaxClassifier(nn.Module):
    def __init__(self, num_seasons=4, num_subtypes=3):
        super(HierarchicalSoftmaxClassifier, self).__init__()
        # Load pre-trained ViT
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Get the feature dimension from the pre-trained model
        feature_dim = self.backbone.heads.head.in_features

        # Remove the classification head
        self.backbone.heads = nn.Identity()

        # Season classifier
        self.season_classifier = nn.Linear(feature_dim, num_seasons)

        # Subtype classifiers (one for each season)
        self.subtype_classifiers = nn.ModuleList([
            nn.Linear(feature_dim, num_subtypes) for _ in range(num_seasons)
        ])

    def forward(self, x):
        # Extract features
        features = self.backbone(x)

        # Predict seasons
        season_logits = self.season_classifier(features)
        season_probs = nn.functional.softmax(season_logits, dim=1)

        # Predict subtypes for each season
        subtype_logits_list = [classifier(features) for classifier in self.subtype_classifiers]
        subtype_probs_list = [nn.functional.softmax(logits, dim=1) for logits in subtype_logits_list]

        # Calculate joint probabilities for all 12 classes
        batch_size = x.size(0)
        joint_probs = torch.zeros(batch_size, 4 * 3).to(x.device)

        for s in range(4):
            for t in range(3):
                idx = s * 3 + t  # Convert to flat index
                joint_probs[:, idx] = season_probs[:, s] * subtype_probs_list[s][:, t]

        return {
            'season_logits': season_logits,
            'season_probs': season_probs,
            'subtype_logits_list': subtype_logits_list,
            'subtype_probs_list': subtype_probs_list,
            'joint_probs': joint_probs
        }


def hierarchical_softmax_loss(outputs, season_targets, subtype_targets=None, full_targets=None):
    """
    Calculate loss using hierarchical softmax.

    Args:
        outputs (dict): Model outputs from forward pass
        season_targets (torch.Tensor): Season labels (0-3)
        subtype_targets (torch.Tensor, optional): Subtype labels (0-2)
        full_targets (torch.Tensor, optional): Combined labels (0-11)

    Returns:
        torch.Tensor: Loss value
    """
    batch_size = season_targets.size(0)
    device = season_targets.device

    # If we have full_targets but not subtype_targets, derive them
    if subtype_targets is None and full_targets is not None:
        season_targets = full_targets // 3
        subtype_targets = full_targets % 3

    # If we have separate season and subtype targets but not full_targets
    if full_targets is None and subtype_targets is not None:
        full_targets = season_targets * 3 + subtype_targets

    # Calculate negative log likelihood of the correct full class
    log_probs = torch.log(outputs['joint_probs'] + 1e-10)
    loss = nn.functional.nll_loss(log_probs, full_targets)

    return loss


def train_with_hierarchical_softmax(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    season_correct = 0
    full_correct = 0
    total = 0

    for inputs, full_targets in dataloader:
        inputs = inputs.to(device)
        full_targets = full_targets.to(device)

        # Derive season and subtype targets
        season_targets = full_targets // 3
        subtype_targets = full_targets % 3

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = hierarchical_softmax_loss(outputs, season_targets, subtype_targets, full_targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)

        # Season accuracy
        _, season_preds = torch.max(outputs['season_probs'], 1)
        season_correct += (season_preds == season_targets).sum().item()

        # Full 12-class accuracy
        _, full_preds = torch.max(outputs['joint_probs'], 1)
        full_correct += (full_preds == full_targets).sum().item()

        total += inputs.size(0)

    return running_loss / total, season_correct / total, full_correct / total