 TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations

This repository provides the official implementation of **TSS-LCD**, a two-stage deep learning framework that jointly captures temporal–spectral–spatial (TSS) dependencies and performs latent diffusion-based generation for fine-grained multi-band spectrum RSS prediction under incomplete historical observations.

This implementation corresponds to our paper:

> **TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations**

---

## Algorithm Overview
TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations

This repository provides the official implementation of TSS-LCD, a two-stage deep learning framework that jointly captures temporal–spectral–spatial (TSS) dependencies and performs latent diffusion–based generation for fine-grained multi-band spectrum RSS prediction under incomplete historical observations.

This implementation corresponds to our paper:

TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations

Algorithm Overview
git

Quick Start
1. Installation
pip install -r requirements.txt

2. Dataset Preparation

Place your dataset CSV file here:

./dataset/merged_power_data_sub6GHz_avg_per_minute.csv


Expected CSV format (RSS in dBm):

freq_1, freq_2, ..., freq_750
-92.1, -90.4, ..., -101.3
...

3. Training

This will train:

Context Transformer Autoencoder

Future Autoencoder

Latent Conditional Diffusion Model

Run:

python train.py --mode train --csv ./dataset/merged_power_data_sub6GHz_avg_per_minute.csv


Model weights will be saved in:

./checkpoints/
    ae_ctx.pth
    ae_fut.pth
    diffusion.pth

4. Testing / Inference

Run:

python train.py --mode test --csv ./dataset/merged_power_data_sub6GHz_avg_per_minute.csv


The script will generate:

./results/spectrogram_comparison.png   # Heatmap: ground truth vs prediction
./results/band_curves.png              # Band-wise RSS comparison
./pred_matours.csv                     # Predicted future RSS
./gt_mat.csv                            # Ground-truth future RSS

Project Structure
.
├── Context2CondNew.py        # TSS-guided context autoencoder (historical part)
├── F2Cnet.py                 # Future autoencoder (latent space)
├── NoiseNet.py               # Diffusion model and noise estimation network
├── train.py                  # Training & inference pipeline
├── data_utils.py             # Data preprocessing and helper functions
├── DataSetPrepare.py         # Dataset utilities
├── dataset/                  # Input CSV data
├── results/                  # Generated figures and prediction files
├── checkpoints/              # Saved model weights
└── requirements.txt          # Python dependencies
