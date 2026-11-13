# Cervical Cell Anomaly Detection Using a Hybrid β-VAE and WGAN-GP Framework

## Overview

This repository contains the research implementation of a hybrid deep learning framework for cervical cell anomaly detection using the SIPaKMeD dataset.

The proposed framework combines a Variational Autoencoder (β-VAE) with a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP).

The project also includes standalone VAE and GAN implementations for comparative evaluation.

## Research Objectives

- Develop a hybrid generative deep learning framework for cervical cell anomaly detection.
- Investigate the application of β-VAE and WGAN-GP in medical image analysis.
- Compare the hybrid framework with standalone VAE and GAN models.
- Evaluate model performance using appropriate experimental metrics.

## Dataset

**SIPaKMeD – Single-cell Pap Smear Image Dataset**

The dataset is not included in this repository. Users must obtain the dataset from its original source and configure the corresponding dataset path before running the code.

## Models

This repository implements and evaluates four deep learning approaches for cervical cell anomaly detection:

1. **Hybrid β-VAE + WGAN-GP (Proposed Model):** A hybrid framework combining variational representation learning with adversarial training.
2. **Standalone β-VAE:** A variational autoencoder baseline without an adversarial component.
3. **Standalone WGAN-GP:** A Wasserstein GAN with gradient penalty, used as an independent adversarial baseline.
4. **CBiGAN:** A conditional bidirectional generative adversarial network baseline.

All four approaches are evaluated using a consistent experimental framework, including anomaly detection metrics and statistical comparisons.

## Implementation

The project is implemented in Python using PyTorch and other scientific Python libraries.

Detailed installation instructions, experimental configurations, and execution guidelines will be provided as the repository is organized.

## Repository Status

The research code is being organized and documented for reproducibility and scientific reference.

## Citation

Citation information and a permanent DOI will be added after the first archived release.

## License

License information will be added after the licensing conditions of the implementation and dataset have been reviewed.
