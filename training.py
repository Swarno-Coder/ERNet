import torch.optim as optim
from callback import EarlyStopping
from evaluate import evaluate_model
from dataset_preprocessing import get_dataloader
from model import ERNetModel
import torch
import torch.nn as nn
import json

history = dict()
def train_model(model, train_loader, valid_loader, optimizer, criterion, num_epochs=50, lr=1e-4, device='cuda'):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    early_stopping = EarlyStopping(patience=60, verbose=True)
    best_valid_loss = float('inf')
    best_dice = 0
    best_iou = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        valid_loss, valid_metrics = evaluate_model(model, valid_loader, criterion, device)
        history[f"epoch_{epoch}"] = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_metrics": valid_metrics.tolist(),
        }
        # Example usage with validation results
        # if (epoch + 1) % 10 == 0: visualize_results(model, valid_loader, device=device, num_samples=5)

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}')
        print(f'Validation Metrics: Accuracy={valid_metrics[0]:.4f}, Precision={valid_metrics[1]:.4f}, Recall={valid_metrics[2]:.4f}, '
              f'Specificity={valid_metrics[3]:.4f}, Sensitivity={valid_metrics[4]:.4f}, Dice={valid_metrics[5]:.4f}, IoU={valid_metrics[6]:.4f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), '/models/best_model.pth')
        if valid_metrics[5] > best_dice:
            best_dice = valid_metrics[5]
            torch.save(model.state_dict(), '/models/best_dice_model.pth')
        if valid_metrics[6] > best_iou:
            best_iou = valid_metrics[6]
            torch.save(model.state_dict(), '/models/best_iou_model.pth')
        # Step the scheduler
        scheduler.step(valid_loss)

        # Check for early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        torch.save(model.state_dict(), '/models/unet_model.pth')
    print("Training complete. Best model saved as 'best_model.pth'")
    return model, history

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ERNetModel(pretrained=True).to(device)
    data = get_dataloader(8)
    m, h = train_model(model, data[0], data[1], optim.Adam(model.parameters(), lr=1e-4), nn.BCELoss(), 100, 1e-4, device)

    with open('/models/history_ERNet.json', 'w') as f:
        json.dump(history, f)