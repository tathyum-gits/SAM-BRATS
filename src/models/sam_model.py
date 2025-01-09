import os
import torch
import torch.nn as nn
import urllib.request
from segment_anything import sam_model_registry

class CustomSAMHead(nn.Module):
    """Custom SAM model with a UNet-like decoder."""
    
    def __init__(self, sam_model):
        super(CustomSAMHead, self).__init__()
        self.sam_model = sam_model
        
        # UNet-like upsampling decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, images):
        """Forward pass."""
        # images shape: [B, 3, 1024, 1024]
        features = self.sam_model.image_encoder(images)
        masks = self.decoder(features)  # [B, 1, H, W]
        return masks

def get_sam_model(config):
    """Initialize and return the SAM model."""
    # Download checkpoint if it doesn't exist
    os.makedirs(os.path.dirname(config['model']['checkpoint_path']), exist_ok=True)
    
    if not os.path.exists(config['model']['checkpoint_path']):
        print("Downloading SAM checkpoint...")
        urllib.request.urlretrieve(
            config['model']['checkpoint_url'],
            config['model']['checkpoint_path']
        )
        print("Checkpoint downloaded!")

    # Load pretrained SAM
    sam = sam_model_registry[config['model']['type']](
        checkpoint=config['model']['checkpoint_path']
    )
    
    # Freeze the backbone
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Create custom model
    custom_sam = CustomSAMHead(sam).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    return custom_sam
