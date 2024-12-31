from model import ERNetModel
import torch
import torch.nn as nn
import argparse
from evaluate import evaluate_model
from dataset_preprocessing import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = ERNetModel(pretrained=True).to(device)
best_model.load_state_dict(torch.load('/content/best_model.pth', weights_only=True))

test_loss, test_metrics = evaluate_model(best_model, get_dataloader(8)[2], nn.BCELoss(), 'cuda')
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Metrics: Accuracy={test_metrics[0]:.4f}, Precision={test_metrics[1]:.4f}, Recall={test_metrics[2]:.4f}, '
      f'Specificity={test_metrics[3]:.4f}, Sensitivity={test_metrics[4]:.4f}, Dice={test_metrics[5]:.4f}, IoU={test_metrics[6]:.4f}')
# Example usage with test results
# visualize_results(best_model, test_loader, device='cuda', num_samples=5)