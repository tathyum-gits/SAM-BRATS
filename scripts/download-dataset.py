import os
import h5py
import numpy as np
from tqdm import tqdm
import logging
import argparse
import yaml
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def download_dataset(dataset, output_dir):
    """Download dataset from Kaggle."""
    try:
        api = KaggleApi()
        api.authenticate()
        
        logging.info(f"Downloading dataset: {dataset}")
        api.dataset_download_files(dataset, path=output_dir, unzip=True)
        logging.info("Dataset downloaded successfully!")
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        raise

def combine_image_channels(h5_file_path):
    """Combine image channels by taking the mean."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        image = h5_file['image'][:]  # Shape: (240, 240, 4)
        combined_image = np.mean(image, axis=-1)
    return combined_image

def combine_masks(h5_file_path):
    """Combine mask channels by taking the maximum."""
    with h5py.File(h5_file_path, 'r') as h5_file:
        mask = h5_file['mask'][:]  # Shape: (240, 240, 3)
        combined_mask = np.maximum.reduce(mask, axis=-1)
    return combined_mask

def process_and_save(file_path, output_dir_images, output_dir_masks):
    """Process a single file and save results."""
    try:
        combined_image = combine_image_channels(file_path)
        combined_mask = combine_masks(file_path)

        # Create output filename
        file_name = Path(file_path).stem
        np.save(os.path.join(output_dir_images, f"{file_name}_image.npy"), combined_image)
        np.save(os.path.join(output_dir_masks, f"{file_name}_mask.npy"), combined_mask)
        
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        raise

def process_dataset(data_dir, output_dir_images, output_dir_masks):
    """Process all files in the dataset."""
    # Create output directories
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_masks, exist_ok=True)

    # Get list of H5 files
    h5_files = list(Path(data_dir).glob('*.h5'))
    
    if not h5_files:
        raise ValueError(f"No .h5 files found in {data_dir}")

    logging.info(f"Found {len(h5_files)} files to process")
    
    # Process each file
    for file_path in tqdm(h5_files, desc="Processing files"):
        process_and_save(str(file_path), output_dir_images, output_dir_masks)

def main(config_path):
    """Main function for dataset download and processing."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create data directory
    os.makedirs(config['paths']['data_dir'], exist_ok=True)

    try:
        # Download dataset
        dataset_name = "awsaf49/brats2020-training-data"
        download_dataset(dataset_name, config['paths']['data_dir'])

        # Set paths for processing
        data_dir = Path(config['paths']['data_dir']) / 'BraTS2020_training_data/content/data'
        
        # Process the dataset
        process_dataset(
            data_dir=data_dir,
            output_dir_images=config['paths']['image_dir'],
            output_dir_masks=config['paths']['mask_dir']
        )
        
        logging.info("Dataset processing completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process BraTS dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()

    main(args.config)
