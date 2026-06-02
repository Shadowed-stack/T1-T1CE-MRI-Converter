# MRI Modality Synthesis using U-Net

## Overview

This project implements a deep learning approach for MRI modality synthesis, where a T1-weighted MRI image is used to generate the corresponding T1 Contrast-Enhanced (T1CE) MRI image.

The model is based on a lightweight 2D U-Net architecture trained on the BraTS 2020 dataset. The objective is to learn the mapping between T1 and T1CE modalities and generate synthetic contrast-enhanced images from non-contrast MRI scans.

---

## Features

- T1 → T1CE MRI synthesis
- U-Net based deep learning architecture
- BraTS 2020 dataset support
- Automated dataset download
- Training and validation pipeline
- Quantitative evaluation metrics
- Medical image preprocessing
- PyTorch implementation

---

## Project Structure

text MRI-T1-to-T1CE/ │ ├── dataset/ │ ├── train.py ├── test.py ├── download_dataset.py │ ├── t1_to_t1ce_model.pth │ ├── requirements.txt └── README.md 

---

## Dataset

This project uses the BraTS 2020 Training Dataset, a benchmark dataset widely used for brain tumor analysis and medical image processing research.

Each sample contains four MRI modalities:

| Channel | Modality |
|----------|----------|
| 0 | T1 |
| 1 | T1CE |
| 2 | T2 |
| 3 | FLAIR |

For this project:

- Input: T1
- Target: T1CE

Each .h5 sample contains:

python image.shape = (H, W, 4) 

where:

- H = image height
- W = image width
- 4 = number of MRI modalities

---

## Installation

Clone the repository:

bash git clone https://github.com/Shadowed-stack/T1-T1CE-MRI-Converter.git  cd T1-T1CE-MRI-Converter

Install dependencies:

bash pip install torch torchvision pip install numpy matplotlib h5py pip install scikit-image scipy pip install kagglehub tqdm 

---

## Download Dataset

Run:

bash python download_dataset.py 

The dataset will be downloaded automatically and stored inside:

text dataset/ 

---

## Model Architecture

The proposed network is a 2D U-Net consisting of:

### Encoder

- Block(1 → 32)
- MaxPool
- Block(32 → 64)
- MaxPool
- Block(64 → 128)
- MaxPool

### Bottleneck

- Block(128 → 128)

### Decoder

- Upsample + Skip Connection
- Block(256 → 128)

- Upsample + Skip Connection
- Block(192 → 64)

- Upsample + Skip Connection
- Block(96 → 32)

### Output Layer

- 1 × 1 Convolution
- Single-channel synthetic T1CE image

### Components Used

- Convolution Layers
- Group Normalization
- GELU Activation
- Bilinear Upsampling
- Skip Connections

---

## Training

Train the model using:

bash python train.py 

Default configuration:

| Parameter | Value |
|------------|--------|
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |

During training:

- Training loss is monitored
- Validation loss is computed
- Best model is automatically saved

Output:

text t1_to_t1ce_model.pth 

---

## Testing

Evaluate a trained model using:

bash python test.py 

The testing script:

1. Loads the trained model
2. Reads a T1 MRI slice
3. Generates a synthetic T1CE image
4. Compares it against the real T1CE image
5. Computes quantitative evaluation metrics

---

## Evaluation Metrics

The generated T1CE image is compared with the corresponding ground-truth T1CE image using:

### Error Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Normalized RMSE (NRMSE)

### Image Quality Metrics

- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index Measure (SSIM)

### Statistical Metrics

- Pearson Correlation Coefficient

### Region-Based Metrics

- Dice Score
- Intersection over Union (IoU)
- Precision
- Recall
- F1 Score
- Accuracy

---

## Sample Results

Example evaluation results obtained from the trained model:

| Metric | Value |
|----------|----------|
| MSE | 0.002851 |
| RMSE | 0.053399 |
| MAE | 0.030796 |
| NRMSE | 0.152741 |
| PSNR | 25.4494 dB |
| SSIM | 0.566295 |
| Pearson Correlation | 0.993289 |
| Dice Score | 0.949591 |
| IoU | 0.904021 |
| Precision | 0.915684 |
| Recall | 0.986106 |
| F1 Score | 0.949591 |
| Accuracy | 0.975799 |

---

## Results Discussion

The model demonstrates strong performance in capturing enhancement patterns and intensity distributions between T1 and T1CE modalities.

Key observations:

- Very high Pearson correlation indicates strong intensity consistency.
- High Dice and IoU scores demonstrate accurate enhancement region identification.
- Low MSE and MAE indicate good pixel-level reconstruction.
- Moderate SSIM suggests that some fine structural details are lost during synthesis.

Overall, the model successfully learns the T1-to-T1CE mapping and generates visually meaningful synthetic contrast-enhanced MRI images.

---

## Applications

Potential applications include:

- Missing MRI modality completion
- Medical image reconstruction
- Brain tumor imaging research
- Data augmentation for medical datasets
- Multimodal MRI analysis
- Preprocessing for segmentation models

---

## Future Improvements

Possible future extensions include:

- 3D U-Net architectures
- Multi-modal input (T1 + T2 + FLAIR)
- SSIM-based loss functions
- Adversarial training using GANs
- Perceptual loss functions
- Tumor-aware patch extraction
- Sliding-window inference
- Transformer-based medical image synthesis

---

## Disclaimer

This project is intended for educational and research purposes only.

The generated T1CE images are synthetic approximations and should not be used for clinical diagnosis, treatment planning, or medical decision-making.

---

## References

1. Ronneberger, O., Fischer, P., Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

2. Menze, B. H., et al. The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS). IEEE Transactions on Medical Imaging.

3. Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE TIP 2004.

4. Goodfellow, I., et al. Generative Adversarial Networks. NeurIPS 2014.

5. BraTS 2020 Challenge Dataset.

6. PyTorch Documentation.
