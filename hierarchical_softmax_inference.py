def inference(model, x, return_probabilities=False):
    """
    Perform inference with the hierarchical model.
    
    Args:
        model (HierarchicalSoftmaxClassifier): The trained model
        x (torch.Tensor): Input image batch
        return_probabilities (bool): If True, return probabilities along with predictions
    
    Returns:
        dict: Dictionary containing predictions and optionally probabilities
    """
    # Set model to evaluation mode
    model.eval()
    
    # Move input to the same device as model if needed
    device = next(model.parameters()).device
    x = x.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(x)
        
        # Get season predictions
        season_probs = outputs['season_probs']
        _, season_preds = torch.max(season_probs, dim=1)
        
        # For each example, get subtype prediction from the predicted season's classifier
        batch_size = x.size(0)
        subtype_preds = torch.zeros(batch_size, dtype=torch.long, device=device)
        subtype_probs = torch.zeros(batch_size, 3, device=device)
        
        for i in range(batch_size):
            pred_season = season_preds[i].item()
            subtype_prob = outputs['subtype_probs_list'][pred_season][i]
            _, subtype_pred = torch.max(subtype_prob, dim=0)
            
            subtype_preds[i] = subtype_pred
            subtype_probs[i] = subtype_prob
        
        # Calculate full class predictions (0-11)
        full_preds = season_preds * 3 + subtype_preds
        
        # Prepare result dictionary
        result = {
            'season_preds': season_preds,
            'subtype_preds': subtype_preds,
            'full_preds': full_preds
        }
        
        if return_probabilities:
            result['season_probs'] = season_probs
            result['subtype_probs'] = subtype_probs
            result['joint_probs'] = outputs['joint_probs']
        
        return result


def predict_color_season(model, img, class_names=None):
    """
    Predict the color season for a single image.
    
    Args:
        model (HierarchicalSoftmaxClassifier): The trained model
        img (torch.Tensor): A single preprocessed image tensor [1, C, H, W]
        class_names (list, optional): List of class names for the 12 seasons
    
    Returns:
        dict: Prediction information
    """
    # Default class names if not provided
    if class_names is None:
        seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
        subtypes = ['Light', 'True', 'Deep']
        class_names = [f"{season} {subtype}" for season in seasons for subtype in subtypes]
    
    # Make sure img is a batch
    if img.dim() == 3:
        img = img.unsqueeze(0)
    
    # Get predictions
    result = inference(model, img, return_probabilities=True)
    
    # For a single image, get the first item in each tensor
    season_idx = result['season_preds'][0].item()
    subtype_idx = result['subtype_preds'][0].item()
    full_idx = result['full_preds'][0].item()
    
    # Get probabilities
    season_prob = result['season_probs'][0, season_idx].item()
    subtype_prob = result['subtype_probs'][0, subtype_idx].item()
    joint_prob = result['joint_probs'][0, full_idx].item()
    
    # Format the results
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    subtypes = ['Light', 'True', 'Deep']
    
    prediction = {
        'predicted_class': class_names[full_idx],
        'season': seasons[season_idx],
        'subtype': subtypes[subtype_idx],
        'confidence': {
            'season_confidence': f"{season_prob:.2%}",
            'subtype_confidence': f"{subtype_prob:.2%}",
            'overall_confidence': f"{joint_prob:.2%}"
        }
    }
    
    return prediction