import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_augmentation():
    """Returns the augmentation pipeline."""
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=90, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])

def preprocess_image(image_path, mask_path, size=1024):
    """Preprocess a single image-mask pair."""
    image = np.load(image_path)
    mask = np.load(mask_path)

    # Resize
    resized_image = cv2.resize(image, (size, size))
    resized_mask = cv2.resize(mask, (size, size))

    # Normalize image
    if resized_image.max() > 0:
        resized_image = resized_image / resized_image.max()

    # Convert to 3 channels if needed
    if resized_image.ndim == 2:
        resized_image = np.stack([resized_image] * 3, axis=-1)
    elif resized_image.shape[0] == 4:  # If 4 channels, keep only first 3
        resized_image = resized_image[:3, :, :].transpose(1, 2, 0)

    # Apply augmentations
    augmented = get_augmentation()(image=resized_image, mask=resized_mask)
    augmented_image = augmented['image']
    augmented_mask = augmented['mask']

    return augmented_image.float(), augmented_mask.unsqueeze(0).float()

class BraTSDataset(Dataset):
    """Dataset class for BraTS segmentation."""
    
    def __init__(self, image_dir, mask_dir):
        """Initialize dataset."""
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.npy')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
            if f.endswith('.npy')
        ])

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        image, mask = preprocess_image(self.image_paths[idx], self.mask_paths[idx])
        return image, mask

def get_dataloaders(config):
    """Create train, validation, and test dataloaders."""
    # Create dataset
    dataset = BraTSDataset(config['paths']['image_dir'], config['paths']['mask_dir'])
    dataset_size = len(dataset)

    # Calculate sizes
    train_size = int(config['data']['train_fraction'] * dataset_size)
    val_size = int(config['data']['val_fraction'] * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(config['training']['seed'])
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
