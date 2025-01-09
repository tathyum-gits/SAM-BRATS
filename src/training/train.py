import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import csv
from .losses import CombinedLoss, dice_score, precision, mae
from ..utils.visualization import visualize_prediction

def save_checkpoint(model, optimizer, epoch, iteration, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_dir):
    """Load model checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")
    
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['iteration']
    else:
        print("No checkpoint found, starting from scratch.")
        return 0, 0

def validate_model(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    dice_scores = []
    precisions = []
    mae_values = []

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            preds = model(images)
            preds = F.interpolate(preds, size=masks.shape[-2:], 
                                mode='bilinear', align_corners=False)
            
            # Calculate loss
            loss = criterion(preds, masks)
            val_loss += loss.item()

            # Calculate metrics
            dice_val = dice_score(preds, masks)
            prec_val = precision(preds, masks)
            mae_val = mae(preds, masks)

            dice_scores.append(dice_val.item())
            precisions.append(prec_val)
            mae_values.append(mae_val)

    # Calculate averages
    avg_loss = val_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)
    avg_precision = np.mean(precisions)
    avg_mae = np.mean(mae_values)

    return avg_loss, avg_dice, avg_precision, avg_mae

def append_metrics_to_csv(file_path, epoch, train_metrics, val_metrics=None):
    """Append training and validation metrics to CSV file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            header = [
                "Epoch",
                "Train_Loss", "Train_Dice", "Train_Precision", "Train_MAE",
                "Val_Loss", "Val_Dice", "Val_Precision", "Val_MAE"
            ]
            writer.writerow(header)

        row = [epoch]
        # Add training metrics
        row.extend([f"{v:.6f}" for v in train_metrics])
        # Add validation metrics if available
        if val_metrics:
            row.extend([f"{v:.6f}" for v in val_metrics])
        else:
            row.extend([""] * 4)  # Empty placeholders for validation metrics
        
        writer.writerow(row)

def train(model, config, train_loader, val_loader, optimizer):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize criterion and scaler for mixed precision
    criterion = CombinedLoss(config)
    scaler = GradScaler(enabled=True)
    
    # Load checkpoint if exists
    start_epoch, start_iteration = load_checkpoint(
        model, optimizer, config['paths']['checkpoint_dir']
    )
    
    # Training loop
    for epoch in range(start_epoch, config['training']['num_epochs']):
        model.train()
        epoch_loss = 0.0
        dice_scores = []
        precision_scores = []
        mae_scores = []

        # Initialize progress bar
        progress_bar = tqdm(train_loader, 
                          desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}", 
                          leave=True)

        for iteration, (images, masks) in enumerate(progress_bar):
            # Skip iterations if resuming mid-epoch
            if epoch == start_epoch and iteration < start_iteration:
                continue

            images, masks = images.to(device), masks.to(device)

            # Mixed precision training
            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(images)
                preds = F.interpolate(preds, size=masks.shape[-2:], 
                                    mode='bilinear', align_corners=False)
                loss = criterion(preds, masks)

            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate metrics
            with torch.no_grad():
                d_score = dice_score(preds, masks)
                prec = precision(preds, masks)
                mae_val = mae(preds, masks)

            # Update metrics
            epoch_loss += loss.item()
            dice_scores.append(d_score.item())
            precision_scores.append(prec)
            mae_scores.append(mae_val)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{d_score.item():.4f}"
            })

            # Save checkpoint at intervals
            if (iteration + 1) % config['training']['save_interval'] == 0:
                save_checkpoint(model, optimizer, epoch, iteration, 
                              config['paths']['checkpoint_dir'])

        # Calculate training metrics
        train_metrics = [
            epoch_loss / len(train_loader),  # avg_loss
            np.mean(dice_scores),            # avg_dice
            np.mean(precision_scores),       # avg_precision
            np.mean(mae_scores)              # avg_mae
        ]

        # Validate
        print("\nRunning validation...")
        val_metrics = validate_model(model, val_loader, criterion, device)

        # Log metrics
        append_metrics_to_csv(
            config['paths']['metrics_log'],
            epoch + 1,
            train_metrics,
            val_metrics
        )

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']} Summary:")
        print(f"  [Train] Loss: {train_metrics[0]:.4f}, "
              f"Dice: {train_metrics[1]:.4f}, "
              f"Precision: {train_metrics[2]:.4f}, "
              f"MAE: {train_metrics[3]:.4f}")
        print(f"  [Val]   Loss: {val_metrics[0]:.4f}, "
              f"Dice: {val_metrics[1]:.4f}, "
              f"Precision: {val_metrics[2]:.4f}, "
              f"MAE: {val_metrics[3]:.4f}")

        # Save checkpoint at end of epoch
        save_checkpoint(model, optimizer, epoch+1, 0, 
                       config['paths']['checkpoint_dir'])

        # Visualize predictions
        if config['training']['visualize_indices']:
            visualize_prediction(
                model=model,
                dataset=val_loader.dataset,
                epoch=epoch,
                save_dir=config['paths']['visualization_dir'],
                device=device,
                indices=config['training']['visualize_indices'],
                threshold=0.2
            )

    return model
