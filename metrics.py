def compute_metrics(pred, target, threshold=0.5):
    # Binarize predictions and target
    pred = (pred > threshold).float()
    target = (target > threshold).float()

    TP = (pred * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    sensitivity = recall  # Sensitivity is the same as recall
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)

    return accuracy, precision, recall, specificity, sensitivity, dice, iou
