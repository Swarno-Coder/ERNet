from metrics import compute_metrics
import numpy as np
import torch

# Model Evaluation function
def evaluate_model(model, dataloader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_metrics = np.zeros(7)  # To store accumulated metrics

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            # Compute metrics
            metrics = compute_metrics(outputs, masks)
            total_metrics += np.array(metrics) * images.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_metrics = total_metrics / len(dataloader.dataset)

    return avg_loss, avg_metrics
