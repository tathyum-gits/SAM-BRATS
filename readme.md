# BraTS Segmentation using SAM (Segment Anything Model)

This repository contains code for training a custom Segment Anything Model (SAM) for brain tumor segmentation using the BraTS 2020 dataset.

## Features

- Custom SAM model adapted for brain tumor segmentation
- Data preprocessing pipeline for BraTS dataset
- Training with mixed precision and gradient scaling
- Multiple loss functions (Dice, Boundary, Focal)
- Visualization tools for predictions
- Checkpoint saving and loading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brats-sam.git
cd brats-sam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the BraTS dataset:
```bash
python scripts/download_dataset.py --output_dir ./data
```

## Usage

1. Train the model:
```bash
python scripts/train_model.py
```

2. Monitor training metrics and visualizations in the specified output directories.

## Model Architecture

The model uses SAM's ViT-B backbone with a custom decoder head for segmentation. The architecture includes:
- Frozen SAM image encoder
- Custom UNet-like decoder
- Multiple loss functions for better boundary adherence

## Results

Include your model's performance metrics and sample visualizations here.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BraTS 2020 dataset
- Meta AI's Segment Anything Model (SAM)

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{brats-sam,
  author = {Your Name},
  title = {BraTS Segmentation using SAM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/brats-sam}
}
```
