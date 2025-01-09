import os
import yaml
import torch
import torch.optim as optim
import sys
import argparse
from pathlib import Path

# Add src to python path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from src.data.dataset import get_dataloaders
from src.models.sam_model import get_sam_model
from src.training.train import train
from src.utils.visualization import plot_metrics

def main(config_path):
    """Main training function."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create necessary directories
    for path in config['paths'].values():
        os.makedirs(path, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])

    # Get data loaders
    print("Initializing data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print(f"Dataset sizes: Train: {len(train_loader.dataset)}, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Initialize model
    print("Initializing model...")
    model = get_sam_model(config)
    print("Model initialized successfully!")

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Train model
    print("Starting training...")
    trained_model = train(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer
    )

    # Plot and save metrics
    print("Plotting metrics...")
    metrics_fig = plot_metrics(config['paths']['metrics_log'])
    metrics_fig.savefig(os.path.join(config['paths']['visualization_dir'], 'metrics.png'))

    print("Training completed!")
    return trained_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAM model on BraTS dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()

    main(args.config)
