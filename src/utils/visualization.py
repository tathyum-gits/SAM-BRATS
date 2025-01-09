import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction(
    model,
    dataset,
    epoch,
    save_dir,
    device='cuda',
    indices=None,
    threshold=0.2,
    colormap='gray',
    alpha=0.5
):
    """
    Visualizes model predictions and saves both three-panel and overlay images.
    
    Args:
        model: Trained model for prediction
        dataset: Dataset containing samples
        epoch: Current epoch number
        save_dir: Directory to save visualizations
        device: Device to run inference on
        indices: List of indices to visualize
        threshold: Threshold for binary prediction
        colormap: Colormap for mask visualization
        alpha: Transparency for overlay visualization
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if indices is None:
        indices = [0]

    with torch.no_grad():
        for idx in indices:
            try:
                # Get sample
                image, mask = dataset[idx]
                image_tensor = image.unsqueeze(0).to(device)
                mask_tensor = mask.unsqueeze(0).to(device)

                # Get prediction
                pred = model(image_tensor)
                pred = F.interpolate(pred, size=mask_tensor.shape[-2:],
                                  mode='bilinear', align_corners=False)

                # Convert to numpy
                image_np = image_tensor.squeeze().cpu().numpy()
                mask_np = mask_tensor.squeeze().cpu().numpy()
                pred_np = pred.squeeze().cpu().numpy()

                # Process prediction
                pred_prob = torch.sigmoid(torch.tensor(pred_np)).numpy()
                pred_mask = (pred_prob > threshold).astype('float32')

                # Create save directory for this sample
                sample_dir = os.path.join(save_dir, f"sample_{idx}")
                os.makedirs(sample_dir, exist_ok=True)

                # 1. Three-panel visualization
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                
                # Input image
                if image_np.shape[0] == 3:
                    axs[0].imshow(image_np.transpose(1, 2, 0))
                else:
                    axs[0].imshow(image_np[0], cmap='gray')
                axs[0].set_title("Input Image")
                axs[0].axis('off')

                # Ground truth
                axs[1].imshow(mask_np.squeeze(), cmap=colormap)
                axs[1].set_title("Ground Truth")
                axs[1].axis('off')

                # Prediction
                axs[2].imshow(pred_mask.squeeze(), cmap=colormap)
                axs[2].set_title("Prediction")
                axs[2].axis('off')

                plt.tight_layout()
                three_panel_path = os.path.join(sample_dir, f'three_panel_epoch_{epoch+1}.png')
                plt.savefig(three_panel_path, bbox_inches='tight', dpi=300)
                plt.close(fig)

                # 2. Overlay visualization
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Create overlay
                if image_np.shape[0] == 3:
                    base_img = image_np.transpose(1, 2, 0)
                else:
                    base_img = np.stack([image_np[0]] * 3, axis=-1)
                
                mask_overlay = np.zeros_like(base_img)
                mask_overlay[:, :, 0] = pred_mask.squeeze()  # Red channel for prediction
                
                blended = (1 - alpha) * base_img + alpha * mask_overlay
                
                ax.imshow(blended)
                ax.set_title("Prediction Overlay")
                ax.axis('off')
                
                overlay_path = os.path.join(sample_dir, f'overlay_epoch_{epoch+1}.png')
                plt.savefig(overlay_path, bbox_inches='tight', dpi=300)
                plt.close(fig)

            except Exception as e:
                print(f"Error processing visualization for index {idx}: {str(e)}")

def plot_metrics(metrics_file):
    """
    Plot training and validation metrics from CSV log file.
    
    Args:
        metrics_file: Path to the CSV file containing metrics
    """
    import pandas as pd
    
    # Read metrics
    df = pd.read_csv(metrics_file)
    
    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Loss
    axs[0, 0].plot(df['Epoch'], df['Train_Loss'], label='Train')
    axs[0, 0].plot(df['Epoch'], df['Val_Loss'], label='Validation')
    axs[0, 0].set_title('Loss')
    axs[0, 0].legend()
    
    # Plot Dice Score
    axs[0, 1].plot(df['Epoch'], df['Train_Dice'], label='Train')
    axs[0, 1].plot(df['Epoch'], df['Val_Dice'], label='Validation')
    axs[0, 1].set_title('Dice Score')
    axs[0, 1].legend()
    
    # Plot Precision
    axs[1, 0].plot(df['Epoch'], df['Train_Precision'], label='Train')
    axs[1, 0].plot(df['Epoch'], df['Val_Precision'], label='Validation')
    axs[1, 0].set_title('Precision')
    axs[1, 0].legend()
    
    # Plot MAE
    axs[1, 1].plot(df['Epoch'], df['Train_MAE'], label='Train')
    axs[1, 1].plot(df['Epoch'], df['Val_MAE'], label='Validation')
    axs[1, 1].set_title('MAE')
    axs[1, 1].legend()
    
    plt.tight_layout()
    return fig
