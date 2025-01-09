from setuptools import setup, find_packages

setup(
    name="brats-sam",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.65.0",
        "albumentations>=1.3.0",
        "h5py>=3.8.0",
        "kaggle>=1.5.13",
        "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        "scikit-learn>=1.0.0",
        "scipy>=1.9.0",
        "pyyaml>=5.1",
    ],
    author="Tathyum",
    author_email="tathyum01@gmail.com",
    description="Brain tumor segmentation using SAM",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tathyum-gits/SAM-BRATS",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
