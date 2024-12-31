import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
import cv2
import numpy as np


benign_dir = 'dataset/Dataset_BUSI_with_GT/benign'
malignant_dir = 'dataset/Dataset_BUSI_with_GT/malignant'

# Function to get image-mask pairs
def get_image_mask_pairs(directory, label):
    images = []
    masks = []
    for file in os.listdir(directory):
        if '_mask' not in file:
            image_path = os.path.join(directory, file)
            mask_path = image_path.replace('.png', '_mask.png')
            if os.path.exists(mask_path):
                images.append(image_path)
                masks.append(mask_path)
    return images, masks

# Combine benign and malignant images and masks
benign_images, benign_masks = get_image_mask_pairs(benign_dir, 'benign')
malignant_images, malignant_masks = get_image_mask_pairs(malignant_dir, 'malignant')

all_images = benign_images + malignant_images
all_masks = benign_masks + malignant_masks

# Ensure the images and masks are paired correctly
assert len(all_images) == len(all_masks), "Number of images and masks must be the same"
assert all([os.path.basename(img).replace('.png', '') == os.path.basename(msk).replace('_mask.png', '') for img, msk in zip(all_images, all_masks)]), "Images and masks are not matched"

# Shuffle the data
combined = list(zip(all_images, all_masks))
all_images, all_masks = zip(*combined)

# Split the data into train (80%), and temp (20%) which we'll split again
train_images, temp_images, train_masks, temp_masks = train_test_split(all_images, all_masks, test_size=0.3, random_state=42)

# Split the temp dataset into validation (50% of temp) and test (50% of temp)
valid_images, test_images, valid_masks, test_masks = train_test_split(temp_images, temp_masks, test_size=0.5, random_state=42)

# Check the splits
print(f"Training set size: {len(train_images), len(train_masks)}")
print(f"Validation set size: {len(valid_images), len(valid_masks)}")
print(f"Test set size: {len(test_images), len(test_masks)}")

# Define dataset class
class CustomDatasetTrainVal(Dataset):
    def __init__(self, image_paths, target_paths, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        target = cv2.imread(self.target_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image, mask=target)
            image = augmented['image']
            target = augmented['mask']
        image = torch.from_numpy(image).permute(2,0,1) / 255.0
        target = torch.from_numpy(target.astype(np.float32) / 255.0).unsqueeze(0)
        return image, target

# Define transformations
train_transform = A.Compose([
    A.Flip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    #A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(p=0.6),
    #A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
    A.Resize(height=256, width=256),
])
valid_transform = A.Compose([
    A.Resize(height=256, width=256)
])

# Load dataset
train_dataset = CustomDatasetTrainVal(
    image_paths = train_images,
    target_paths = train_masks,
    transform = train_transform
)
val_dataset = CustomDatasetTrainVal(
    image_paths = valid_images,
    target_paths = valid_masks,
    transform = valid_transform
)
test_dataset = CustomDatasetTrainVal(
    image_paths = test_images,
    target_paths = test_masks,
    transform = valid_transform
)
def get_dataloader(batch_size, shuffle=True):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return (train_loader, val_loader, test_loader)
