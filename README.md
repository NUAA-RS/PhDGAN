# PhDGAN - A Physics-Informed Dual-Branch GAN with Gamma-Prior for Dual-Polarization SAR Image Colorization

This repository provides the official PyTorch implementation for the paper PhDGAN. Our framework explicitly incorporates SAR physical scattering properties through a Gamma-Prior Embedding Module (GPEM) to achieve high-fidelity dual-polarization (VV and VH) SAR image colorization.

## 1. Introduction

Dual-polarization SAR image colorization is a challenging task due to the complex physical scattering mechanisms. PhDGAN tackles this fundamental challenge by decoupling texture heterogeneity from SAR intensity and successfully aligning it with natural optical semantics, effectively suppressing perceptual color distortions.

## 2. Environment Setup

To ensure the rigorous reproducibility of our results, we strongly recommend using a virtual environment. The required dependencies include Python, PyTorch, and standard scientific computation libraries such as NumPy, OpenCV, and Scikit-image. You can easily install all necessary packages by running the command below

```bash
pip install -r requirements.txt
```

## 3. Data Preparation and DPSCD Dataset

The Dual-Polarization SAR Colorization Dataset (DPSCD) is constructed from the raw SEN1-2 dataset using our Spectral Correction Module (SCM).

* **SCM Implementation** The core script for generating colorized pseudo-labels is located in `scm.py`. This script demonstrates how to successfully integrate optical color information with SAR radiometric textures without losing radar fidelity.
* **Dataset Download** For convenience and quick reproduction, we provide the fully pre-processed DPSCD dataset ready for training. Please download the dataset directly from `pan.baidu.com/s/1kiXEvBY2wrRPdtWupnQ8-g?pwd=bjtt` (simply copy and paste this link into your browser) and place the unzipped files into the `data/` directory.
* **Custom Dataset Preparation** If you have your own dataset, you can organize it in the corresponding format and use `scm.py` to generate your custom colorized labels. Simply modify the directory variables in the script as shown below to match your local paths

```python
folder1 = 'dataset/SAR'
folder2 = 'dataset/OPT'
output_folder1 = 'output/VV'
output_folder2 = 'output/VH'
```

## 4. Core Modules

The physical prior module (GPEM) is implemented in `generator_gam_deep.py`, which serves as the key component in our architecture for extracting statistical scattering properties. The dual-branch structure ensures independent processing of VV and VH polarizations to preserve their unique scattering characteristics.

## 5. Training and Evaluation

### Training
To start the training process from scratch, simply execute the script below. This will initialize the dual-branch generator and discriminators, applying the Coupled Supervision Loss as explicitly described in the paper. The training logs and periodic checkpoints will be automatically saved in the `checkpoints/` directory.

```bash
python train.py
```

### Testing
To evaluate the model and reproduce the quantitative metrics reported in our paper (including PSNR, SSIM, SAM, NRMSE, and CIEDE2000), please run the testing script and specify the trained checkpoint path.

```bash
python test.py --checkpoint_path checkpoints/best_model.pth
```
