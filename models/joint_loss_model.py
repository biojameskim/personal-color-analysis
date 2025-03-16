import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ColorSeasonClassifier(nn.Module):
    def __init__(self, num_seasons=4, num_subtypes=3):
        super(ColorSeasonClassifier, self).__init__()

        # pre-trained ViT
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Get feature dimension
        feature_dim = self.vit.heads.head.in_features  # This is 768

        # Remove the classification head
        self.vit.heads = nn.Identity()

        # Two classification heads for seasons and subtypes
        self.season_classifier = nn.Linear(feature_dim, num_seasons)
        self.subtype_classifier = nn.Linear(feature_dim, num_subtypes)

        # Classifier for the full 12 classes (Used for evaluation)
        self.full_classifier = nn.Linear(
            feature_dim, num_seasons * num_subtypes)

    def forward(self, x):
        features = self.vit(x)

        # Get predictions
        season_logits = self.season_classifier(features)
        subtype_logits = self.subtype_classifier(features)
        full_logits = self.full_classifier(features)

        # Convert to probabilities
        season_probs = nn.functional.softmax(season_logits, dim=1)
        subtype_probs = nn.functional.softmax(subtype_logits, dim=1)
        full_probs = nn.functional.softmax(full_logits, dim=1)

        return {
            'features': features,
            'season_logits': season_logits,
            'season_probs': season_probs,
            'subtype_logits': subtype_logits,
            'subtype_probs': subtype_probs,
            'full_logits': full_logits,
            'full_probs': full_probs
        }


def val(model, val_loader, criterion, device, alpha=0.4, beta=0.4, gamma=0.2):
    """
    Validation
    """
    val_running_loss = 0.0
    season_correct = 0
    subtype_correct = 0
    full_correct = 0
    total = 0

    model.eval()  # inference mode

    with torch.inference_mode():
        for inputs, full_targets in val_loader:
            inputs, full_targets = inputs.to(device), full_targets.to(device)

            # Derive season and subtype targets
            season_targets = full_targets // 3
            subtype_targets = full_targets % 3

            outputs = model(inputs)
            loss, season_loss, subtype_loss, full_loss = criterion(
                outputs, season_targets, subtype_targets, full_targets,
                alpha=alpha, beta=beta, gamma=gamma
            )
            val_running_loss += loss.item()  # Add loss of this batch to running total

            _, season_preds = torch.max(outputs['season_probs'], 1)
            season_correct += (season_preds == season_targets).sum().item()

            _, subtype_preds = torch.max(outputs['subtype_probs'], 1)
            subtype_correct += (subtype_preds == subtype_targets).sum().item()

            _, full_preds = torch.max(outputs['full_probs'], 1)
            full_correct += (full_preds == full_targets).sum().item()

            total += full_targets.size(0)  # Add batch size to total

    val_running_loss = val_running_loss / len(val_loader)
    val_season_acc = season_correct / total
    val_subtype_acc = subtype_correct / total
    val_full_acc = full_correct / total

    return val_running_loss, val_season_acc, val_subtype_acc, val_full_acc


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, alpha=0.4, beta=0.4, gamma=0.2):
    """
    Train with joint loss
    """
    train_loss_arr = []
    season_acc_arr = []
    subtype_acc_arr = []
    full_acc_arr = []
    val_loss_arr = []
    val_season_acc_arr = []
    val_subtype_acc_arr = []
    val_full_acc_arr = []

    running_loss = 0.0

    for epoch in range(epochs):
        model.train()  # training mode
        running_loss = 0.0
        season_correct = 0
        subtype_correct = 0
        full_correct = 0
        total = 0  # total number of images

        for inputs, full_targets in train_loader:
            inputs, full_targets = inputs.to(device), full_targets.to(device)

            # Derive season and subtype targets
            season_targets = full_targets // 3
            subtype_targets = full_targets % 3

            optimizer.zero_grad()
            outputs = model(inputs)
            loss, season_loss, subtype_loss, full_loss = criterion(
                outputs, season_targets, subtype_targets, full_targets,
                alpha=alpha, beta=beta, gamma=gamma
            )
            loss.backward()
            optimizer.step()

            # Compute metrics
            running_loss += loss.item()  # Add loss of this batch to running total
            total += full_targets.size(0)  # Add batch size to total

            _, season_preds = torch.max(outputs['season_probs'], 1)
            season_correct += (season_preds == season_targets).sum().item()

            _, subtype_preds = torch.max(outputs['subtype_probs'], 1)
            subtype_correct += (subtype_preds == subtype_targets).sum().item()

            _, full_preds = torch.max(outputs['full_probs'], 1)
            full_correct += (full_preds == full_targets).sum().item()

        # average training loss per batch in this epoch
        epoch_train_loss = running_loss / len(train_loader)
        train_loss_arr.append(epoch_train_loss)

        season_acc_arr.append(season_correct / total)
        subtype_acc_arr.append(subtype_correct / total)
        full_acc_arr.append(full_correct / total)

        # Validation
        val_loss, val_season_acc, val_subtype_acc, val_full_acc = val(
            model, val_loader, criterion, device, alpha, beta, gamma)
        val_loss_arr.append(val_loss)
        val_season_acc_arr.append(val_season_acc)
        val_subtype_acc_arr.append(val_subtype_acc)
        val_full_acc_arr.append(val_full_acc)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {full_correct/total:.4f}, Val Acc: {val_full_acc:.4f}")
    results = {
        "train_loss_arr": train_loss_arr,
        "season_acc_arr": season_acc_arr,
        "subtype_acc_arr": subtype_acc_arr,
        "full_acc_arr": full_acc_arr,
        "val_loss_arr": val_loss_arr,
        "val_season_acc_arr": val_season_acc_arr,
        "val_subtype_acc_arr": val_subtype_acc_arr,
        "val_full_acc_arr": val_full_acc_arr
    }
    return results
