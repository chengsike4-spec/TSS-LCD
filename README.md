# TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations

This repository provides the official implementation of **TSS-LCD**, a two-stage deep learning framework that jointly captures temporal–spectral–spatial (TSS) dependencies and performs latent diffusion–based generation for fine-grained multi-band spectrum RSS prediction under incomplete historical observations.

This implementation corresponds to our paper:

**TSS-LCD: A Temporal–Spectral–Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations**

---

## Algorithm Overview

TSS-LCD adopts a two-stage architecture:

1. A Temporal–Spectral–Spatial attention module constructs a unified conditional representation from incomplete historical spectrum observations.  
2. A latent conditional diffusion model generates fine-grained future RSS spectra in a compact latent space and decodes them back to the original spectrum domain.

You can insert the framework figure here (optional):

```markdown
![TSS-LCD Framework](./results/tss_lcd_overview.png)
```

---

## Quick Start

### 1. Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 2. Dataset Preparation

Place your merged spectrum CSV file at:

```bash
./dataset/merged_power_data_sub6GHz_avg_per_minute.csv
```

The expected CSV format (RSS in dBm) is:

```text
freq_1, freq_2, ..., freq_750
-92.1,  -90.4,  ..., -101.3
-93.0,  -89.8,  ..., -100.5
...
```

---

### 3. Training

This will train:

- Context Transformer Autoencoder  
- Future Autoencoder  
- Latent Conditional Diffusion Model  

Run:

```bash
python train.py --mode train --csv ./dataset/merged_power_data_sub6GHz_avg_per_minute.csv
```

Trained model weights will be saved in:

```text
./checkpoints/
    ae_ctx.pth
    ae_fut.pth
    diffusion.pth
```

---

### 4. Testing / Inference

To evaluate the trained model and generate prediction figures:

```bash
python train.py --mode test --csv ./dataset/merged_power_data_sub6GHz_avg_per_minute.csv
```

The script will generate:

```text
./results/spectrogram_comparison.png   # Heatmap: ground truth vs prediction
./results/band_curves.png              # Band-wise RSS comparison curves
./pred_matours.csv                     # Predicted future RSS
./gt_mat.csv                            # Ground-truth future RSS
```

---

## Project Structure

The main files and directories are organized as:

```text
.
├── Context2CondNew.py        # TSS-guided context autoencoder (historical spectrum)
├── F2Cnet.py                 # Future autoencoder in latent space
├── NoiseNet.py               # Latent diffusion model and noise estimation network
├── train.py                  # Training & testing pipeline (AE + Diffusion)
├── data_utils.py             # Data loading, normalization, dataset utilities
├── DataSetPrepare.py         # Additional dataset preprocessing scripts
├── dataset/                  # Input CSV data directory
├── results/                  # Generated plots and prediction outputs
├── checkpoints/              # Saved model weights
└── requirements.txt          # Python dependencies
```

---

## Method Summary

TSS-LCD is designed for spectrum prediction under incomplete observations, and consists of two main stages: TSS-guided condition construction and latent-space diffusion prediction.

---

### Stage 1: Temporal–Spectral–Spatial Attention-Based Condition Construction (TSS-CC)

Historical wideband RSS measurements are often incomplete due to partial-band sensing, hardware limitations, or scheduling policies. The TSS-CC module first builds a powerful conditional representation from such incomplete data.

The historical spectrum tensor is processed by three dedicated branches:

- Temporal Feature Extractor (TemFE): captures long-range temporal dependencies along the time dimension.  
- Spectral Feature Extractor (SpeFE): models correlations across frequency bins, preserving band-wise structure.  
- Spatial Feature Extractor (SpaFE): explores relationships across different sensing locations or nodes.  

Each branch adopts multi-head self-attention to focus on informative entries while being robust to zero-padded or missing values. The three feature maps are then fused by a cross-attention–based Feature Fusion Module (FFM) to obtain a unified conditional feature:

```text
H_fusion  # Temporal–Spectral–Spatial fused condition
```

This fused representation H_fusion is used to guide the subsequent diffusion process.

---

### Stage 2: Latent Conditional Diffusion Spectrum Prediction (LCD-SP)

Instead of predicting RSS directly in the high-dimensional spectrum space, TSS-LCD performs generation in a compact latent space.

The pipeline is:

```text
1. A Latent Space Encoder (LSE) encodes the ground-truth future RSS Y into a latent vector z0.
2. A forward diffusion process gradually adds Gaussian noise to z0, producing noisy latents z_t.
3. A Noise Estimation Network (NEN) takes (z_t, t, H_fusion) as input and predicts the injected noise at each step.
4. During sampling, starting from pure noise, the reverse diffusion process iteratively denoises z_T to reconstruct z0_bar, guided by H_fusion.
5. A Latent Space Decoder (LSD) maps z0_bar back to the spectrum domain, yielding the predicted future RSS Y_bar.
```

By combining latent modeling and conditioned diffusion, TSS-LCD can:

- Preserve fine-grained (≈2–5 dBm) variations in RSS.  
- Avoid over-smoothing effects common in direct regression models.  
- Better handle uncertainty under missing and noisy observations.

---

## Experimental Results

TSS-LCD is evaluated on the AERPAW wideband spectrum dataset with sub-6GHz RSS measurements collected at multiple fixed nodes (CC1, CC2, LW1).

The experimental setting is:

```text
- Input:   50 historical time steps with 25% missing observations
- Output:  10 future time steps of full-band RSS
- Band:    Sub-6GHz, multiple frequency bins (e.g., 750 bins)
- Metrics: MSE, RMSE, MAE, MAPE
```

A comparison with several baselines shows that TSS-LCD achieves the best overall performance:

```text
Method        MSE      RMSE     MAE      MAPE
-------------------------------------------------------
TSS-LCD       0.1621   0.4026   0.2418   0.2032%
STS-PredNet   0.9325   0.9656   0.6810   0.5628%
CSMA          0.6127   0.7827   0.5125   0.4279%
ConvLSTM      1.4158   1.1898   0.8493   0.6985%
```

These results indicate that TSS-LCD:

- Provides more accurate spectrum predictions under incomplete observations.  
- Maintains better robustness when the missing ratio increases.  
- Degrades more slowly as the prediction horizon becomes longer.

---

## Citation

If you find this repository useful, please consider citing our paper:

```bibtex
@article{Cheng2025TSSLCD,
  title   = {TSS-LCD: A Temporal--Spectral--Spatial Guided Latent Conditional Diffusion Model for Spectrum Prediction Under Incomplete Observations},
  author  = {Sike Cheng and Xuanheng Li and Xiangbo Lin and Haichuan Ding and Yi Sun},
  journal = {IEEE Transactions on Cognitive Communications and Networking},
  year    = {2025}
}
```

---

## Contact

If you have any questions or are interested in collaboration, please contact:

```text
Name:   Sike Cheng
Email:  chengsike4@gmail.com
```
