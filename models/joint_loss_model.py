import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt


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
        self.full_classifier = nn.Linear(feature_dim, num_seasons * num_subtypes)

    def forward(self, x):
        features = self.vit(x)

        # Get predictions
        season_logits = self.season_classifier(features)
        subtype_logits = self.subtype_classifier(features)
        full_logits = self.full_classifier(features)

        # Convert to probabilities
        season_probs = F.softmax(season_logits, dim=1)
        subtype_probs = F.softmax(subtype_logits, dim=1)
        full_probs = F.softmax(full_logits, dim=1)

        return {
            'features': features,
            'season_logits': season_logits,
            'season_probs': season_probs,
            'subtype_logits': subtype_logits,
            'subtype_probs': subtype_probs,
            'full_logits': full_logits,
            'full_probs': full_probs
        }


def eval(model, data_loader, criterion, device, alpha=0.4, beta=0.4, gamma=0.2):
    """
    Evaluation function
    """
    model.eval()  # inference mode

    running_loss = 0.0
    season_correct = 0
    subtype_correct = 0
    full_correct = 0
    total = 0

    with torch.inference_mode():
        for inputs, full_targets in data_loader:
            inputs, full_targets = inputs.to(device), full_targets.to(device)

            # Derive season and subtype targets
            season_targets = full_targets // 3
            subtype_targets = full_targets % 3

            outputs = model(inputs)
            loss, _, _, _ = criterion(outputs, season_targets, subtype_targets, full_targets, alpha=alpha, beta=beta, gamma=gamma)
            
            running_loss += loss.item()  # Add loss of this batch to running total
            total += full_targets.size(0)  # Add batch size to total

            _, season_preds = torch.max(outputs['season_probs'], 1)
            season_correct += (season_preds == season_targets).sum().item()

            _, subtype_preds = torch.max(outputs['subtype_probs'], 1)
            subtype_correct += (subtype_preds == subtype_targets).sum().item()

            _, full_preds = torch.max(outputs['full_probs'], 1)
            full_correct += (full_preds == full_targets).sum().item()

    return {
        'loss': running_loss / len(data_loader), # avg loss per batch
        'season_acc': season_correct / total,
        'subtype_acc': subtype_correct / total,
        'full_acc': full_correct / total
    }


def train(model, train_loader, val_loader, optimizer, criterion, device, epochs, alpha=0.4, beta=0.4, gamma=0.2, scheduler=None):
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

    best_val_acc = 0
    best_epoch = 0

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
            loss, _, _, _ = criterion(
                outputs, season_targets, subtype_targets, full_targets,
                alpha=alpha, beta=beta, gamma=gamma
            )
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
        val_metrics = eval(model, val_loader, criterion, device, alpha, beta, gamma)

        val_loss_arr.append(val_metrics['loss'])
        val_season_acc_arr.append(val_metrics['season_acc'])
        val_subtype_acc_arr.append(val_metrics['subtype_acc'])
        val_full_acc_arr.append(val_metrics['full_acc'])

        # Check for best model
        if val_metrics['full_acc'] > best_val_acc:
            best_val_acc = val_metrics['full_acc']
            best_epoch = epoch

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
              f"Train Acc: {full_correct/total:.4f}, Val Full Acc: {val_metrics['full_acc']:.4f}, " 
              f"Val Season Acc: {val_metrics['season_acc']:.4f}, Val Subtype Acc: {val_metrics['subtype_acc']:.4f}")
        
    results = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch + 1,
        "train_losses": train_loss_arr,
        "season_accs": season_acc_arr,
        "subtype_accs": subtype_acc_arr,
        "full_accs": full_acc_arr,
        "val_losses": val_loss_arr,
        "val_season_accs": val_season_acc_arr,
        "val_subtype_accs": val_subtype_acc_arr,
        "val_full_accs": val_full_acc_arr
    }
    return results

def plot_training_history(history, title="Training History", fold=None):
    """
    Plot training and validation metrics including all three accuracy measures
    
    Args:
        history: Dictionary containing training history
        title: Plot title
        fold: Fold number (optional)
    """
    # Create a 2x2 grid of subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot season accuracies
    ax2.plot(history['season_accs'], label='Train Season Acc')
    ax2.plot(history['val_season_accs'], label='Val Season Acc')
    ax2.set_title('Season Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot subtype accuracies
    ax3.plot(history['subtype_accs'], label='Train Subtype Acc')
    ax3.plot(history['val_subtype_accs'], label='Val Subtype Acc')
    ax3.set_title('Subtype Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Plot full accuracies
    ax4.plot(history['full_accs'], label='Train Full Acc')
    ax4.plot(history['val_full_accs'], label='Val Full Acc')
    ax4.set_title('Full Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Set common y-axis limits for the accuracy plots
    max_acc = max([
        max(history['season_accs'] + history['val_season_accs']),
        max(history['subtype_accs'] + history['val_subtype_accs']),
        max(history['full_accs'] + history['val_full_accs'])
    ])
    min_acc = min([
        min(history['season_accs'] + history['val_season_accs']),
        min(history['subtype_accs'] + history['val_subtype_accs']),
        min(history['full_accs'] + history['val_full_accs'])
    ])
    
    padding = (max_acc - min_acc) * 0.1
    for ax in [ax2, ax3, ax4]:
        ax.set_ylim(max(0, min_acc - padding), min(1.0, max_acc + padding))
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save figure
    filename = "training_history"
    if fold is not None:
        filename += f"_fold_{fold}"
    plt.savefig(f"{filename}.png")
    plt.show()

def kfold_crossval(params, train_dataset, val_dataset, model_class, criterion, test_transform, device, n_splits=5):
    """
    Evaluate a model configuration using k-fold cross-validation
    
    Args:
        params: Dictionary of model parameters
        dataset: Dataset to use for evaluation
        device: Device to train on
        n_splits: Number of folds
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating parameters: {params}")

    labels = train_dataset.labels # Need labels for stratification
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Track metrics across folds
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(train_dataset)), labels)):
        print(f"Evaluating fold {fold+1}/{n_splits}")

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler, num_workers=4)
        
        # Create validation dataset with test transforms
        val_dataset.transform = test_transform
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], sampler=val_sampler, num_workers=4)

        model = model_class().to(device) # Reinitialize new model for each fold
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=params['num_epochs'] * len(train_loader))

        train_metrics = train(
            model, train_loader, val_loader, optimizer, criterion, device, 
            params['num_epochs'], alpha=params['alpha'], beta=params['beta'], gamma=params['gamma'], scheduler=scheduler
        )

        # Record metrics for this fold
        fold_metrics.append({
            'fold': fold + 1,
            'best_epoch': train_metrics['best_epoch'],
            'best_val_acc': train_metrics['best_val_acc'],
            'train_losses': train_metrics['train_losses'],
            'season_accs': train_metrics['season_accs'],
            'subtype_accs': train_metrics['subtype_accs'], 
            'full_accs': train_metrics['full_accs'],

            'val_losses': train_metrics['val_losses'],
            'val_season_accs': train_metrics['val_season_accs'],
            'val_subtype_accs': train_metrics['val_subtype_accs'], 
            'val_full_accs': train_metrics['val_full_accs'],
        })

        print(f" Fold {fold+1} best val accuracy: {train_metrics['best_val_acc']:.4f} at epoch {train_metrics['best_epoch']:.4f}")
        plot_training_history(fold_metrics[-1], title="Training History", fold=fold_metrics[-1]['fold'])

    # Calculate average metrics across folds
    avg_best_val_acc = sum(fold['best_val_acc'] for fold in fold_metrics) / len(fold_metrics)
    avg_best_epoch = sum(fold['best_epoch'] for fold in fold_metrics) / len(fold_metrics)
    
    print(f"\nAverage best validation accuracy: {avg_best_val_acc:.4f}")
    print(f"Average best epoch: {avg_best_epoch:.1f}")
    
    return {
        'params': params,
        'fold_metrics': fold_metrics,
        'avg_best_val_acc': avg_best_val_acc,
        'avg_best_epoch': avg_best_epoch
    }

        
def train_full_model():
    """
    TODO: If performance is good, train model on full training data for use in inference.
    """
    pass
        