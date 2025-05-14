def inference_joint_model(model, x, return_probabilities=False):
    """
    Perform inference with the joint loss model.
    
    Args:
        model (ColorSeasonClassifier): The trained model
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
        
        # Get predictions from each classifier
        _, season_preds = torch.max(outputs['season_probs'], dim=1)
        _, subtype_preds = torch.max(outputs['subtype_probs'], dim=1)
        _, full_preds = torch.max(outputs['full_probs'], dim=1)
        
        # Calculate combined predictions from separate season and subtype
        # (This may or may not match the full_preds)
        combined_preds = season_preds * 3 + subtype_preds
        
        # Prepare result dictionary
        result = {
            'season_preds': season_preds,
            'subtype_preds': subtype_preds,
            'full_preds': full_preds,
            'combined_preds': combined_preds
        }
        
        if return_probabilities:
            result['season_probs'] = outputs['season_probs']
            result['subtype_probs'] = outputs['subtype_probs'] 
            result['full_probs'] = outputs['full_probs']
        
        return result
      
def predict_color_season_joint(model, img, class_names=None):
    """
    Predict the color season for a single image using the joint loss model.
    
    Args:
        model (ColorSeasonClassifier): The trained model
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
    result = inference_joint_model(model, img, return_probabilities=True)
    
    # For a single image, get the first item in each tensor
    season_idx = result['season_preds'][0].item()
    subtype_idx = result['subtype_preds'][0].item()
    full_idx = result['full_preds'][0].item()
    combined_idx = result['combined_preds'][0].item()
    
    # Get probabilities
    season_prob = result['season_probs'][0, season_idx].item()
    subtype_prob = result['subtype_probs'][0, subtype_idx].item()
    full_prob = result['full_probs'][0, full_idx].item()
    
    # Format the results
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    subtypes = ['Light', 'True', 'Deep']
    
    prediction = {
        'direct_prediction': class_names[full_idx],
        'combined_prediction': class_names[combined_idx],
        'season': seasons[season_idx],
        'subtype': subtypes[subtype_idx],
        'confidence': {
            'season_confidence': f"{season_prob:.2%}",
            'subtype_confidence': f"{subtype_prob:.2%}",
            'full_class_confidence': f"{full_prob:.2%}"
        },
        'note': "direct_prediction uses the dedicated 12-class classifier, while combined_prediction uses separate season and subtype predictions"
        # During inference, we can either: 
        # Use the dedicated full classifier's prediction directly (full_preds)
        # Combine the season and subtype predictions (combined_preds = season_preds * 3 + subtype_preds)
    }
    
    return prediction