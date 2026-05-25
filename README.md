# PhDGAN: A Physics-Informed Dual-Branch GAN with Gamma-Prior for Dual-Polarization SAR Image Colorization
This repository contains the official PyTorch implementation of the PhDGAN paper. Our framework explicitly incorporates SAR physical scattering properties through a Gamma-Prior Embedding Module (GPEM) to achieve high-fidelity dual-polarization (VV/VH) SAR image colorization.
Labels can be made using scm.py.

# Environment Setup
To ensure the reproducibility of our results, we recommend using a virtual environment. The required dependencies include Python， PyTorch and common libraries such as NumPy, OpenCV, and Scikit-image. You can install all necessary packages by running the command pip install -r requirements.txt provided in this directory.

# Data Preparation and DPSCD Dataset
The Dual-Polarization SAR Colorization Dataset (DPSCD) is constructed from the raw SEN1-2 dataset using our Spectral Correction Module (SCM).
SCM Implementation: The code for generating colorized labels is located in scm.py. This script demonstrates how to integrate optical color information with SAR radiometric textures.
Dataset Download: For convenience, we provide the pre-processed DPSCD dataset ready for training. Please download the dataset from [https://pan.baidu.com/s/1kiXEvBY2wrRPdtWupnQ8-g?pwd=bjtt] and place it in the data/ folder.

# Training the Model
To start the training process from scratch, execute the script python train.py. This script will initialize the dual-branch generator and discriminators, applying the Coupled Supervision Loss as described in the paper. The training logs and periodic checkpoints will be saved in the checkpoints/ directory. And GPEM modle is in generator_gam_deep.py
