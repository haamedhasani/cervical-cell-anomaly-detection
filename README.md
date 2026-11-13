# Cervical Cell Anomaly Detection Using a Hybrid β-VAE and WGAN-GP Framework

A deep learning framework for unsupervised anomaly detection in cervical cell images using the SIPaKMeD dataset.

---

## Overview

This repository contains the research implementation of a deep learning framework for detecting abnormal cervical cells in Pap smear images.

The proposed approach combines a **Beta-Variational Autoencoder (β-VAE)** with a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)**. The framework is designed to learn representations of normal cervical cells and identify anomalous samples through deviations from the learned normal-data distribution.

In addition to the proposed hybrid framework, three standalone generative models are implemented as baselines for comparative evaluation.

The implementation includes data preprocessing, group-aware dataset splitting, model training, anomaly scoring, performance evaluation, statistical analysis, and result visualization.

## Research Objectives

The main objectives of this project are:

- Develop a hybrid generative deep learning framework for cervical cell anomaly detection.
- Learn representative latent features from normal cervical cell images.
- Investigate the combination of variational representation learning and adversarial training.
- Evaluate the proposed framework against three standalone generative baselines.
- Analyze anomaly detection performance using classification metrics and statistical comparisons.
- Provide a reproducible implementation for further research and scientific evaluation.

## Dataset

### SIPaKMeD

The experiments are based on the **SIPaKMeD (Single-cell Pap Smear Image Dataset)**, a publicly available dataset of single-cell images obtained from cervical Pap smear samples.

The implementation organizes the dataset into two main groups:

| Group | Classes |
|---|---|
| Normal | `im_Parabasal`, `im_Superficial-Intermediate` |
| Abnormal | `im_Dyskeratotic`, `im_Koilocytotic`, `im_Metaplastic` |

**Important:** The dataset is not included in this repository. Users must obtain it separately from its original distribution source and comply with the dataset's terms of use.

### Dataset Organization

The data loader supports the following directory structures for each class:

```text
dataset_root/
├── im_Parabasal/
│   └── im_Parabasal/
│       └── CROPPED/
├── im_Superficial-Intermediate/
│   └── im_Superficial-Intermediate/
│       └── CROPPED/
└── ...
```

Alternative supported layouts include:

```text
dataset_root/
├── class_name/
│   └── CROPPED/
└── ...
```

and a flat class directory containing image files.

The current file-discovery implementation searches for `.bmp`, `.png`, and `.jpg` images.

## Methodology

### Data Preprocessing

The image preprocessing pipeline includes:

1. Loading image files from the dataset directories.
2. Converting images to RGB format.
3. Resizing images to 224 × 224 pixels.
4. Applying Gaussian blur with a 3 × 3 kernel.
5. Normalizing pixel values to the [0, 1] range.
6. Applying light data augmentation during training.

### Group-Aware Data Splitting

To reduce potential information leakage between related image samples, the implementation uses `GroupShuffleSplit`.

The dataset is divided as follows:

| Data group | Training | Validation | Testing |
|---|---:|---:|---:|
| Normal images | 70% | 15% | 15% |
| Abnormal images | 60% | 20% | 20% |

The grouping information is inferred from image filenames. The implementation checks that inferred groups do not overlap across the corresponding splits.

**Training strategy:** Only the normal training images are used for model optimization. Normal and abnormal images are included in the validation and test evaluation sets.

## Models

Four generative deep learning approaches are implemented and evaluated in this repository.

### 1. Hybrid β-VAE + WGAN-GP — Proposed Model

The proposed framework integrates variational representation learning and adversarial learning.

Its objective combines:

- Reconstruction loss
- KL divergence regularization
- Adversarial loss
- Feature matching loss
- Wasserstein critic training with gradient penalty

The framework aims to learn a meaningful latent representation of normal cervical cell images while incorporating adversarial learning.

### 2. Standalone β-VAE

A standalone Beta-Variational Autoencoder is implemented as a baseline to evaluate the contribution of variational representation learning without the adversarial component.

### 3. Standalone WGAN-GP

A standalone Wasserstein Generative Adversarial Network with Gradient Penalty is used as an adversarial baseline.

### 4. CBiGAN

A bidirectional generative adversarial network baseline is included for comparative evaluation, following the implementation's referenced CBiGAN formulation.

## Experimental Configuration

The main experimental configuration is defined in `src/main.py`.

| Parameter | Value |
|---|---:|
| Input image size | 224 × 224 |
| Batch size | 16 |
| Latent dimension | 64 |
| Maximum training epochs | 40 |
| Learning rate | 0.0001 |
| Critic learning rate | 0.0001 |
| Weight decay | 0.0005 |
| Random seed | 42 |
| Gradient penalty coefficient | 10.0 |
| Number of critic steps | 5 |

The hybrid model also uses a β-VAE warm-up schedule and a separate adversarial warm-up period.

Early stopping is implemented to help control training and model selection.

These values represent the current code configuration and may be modified for future experiments.

## Anomaly Detection and Evaluation

The implementation includes latent-space analysis and anomaly scoring based on the learned representations.

Gaussian Mixture Models (GMM) and Mahalanobis-distance-based calculations are incorporated into the anomaly detection pipeline.

The evaluation process includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- Confusion matrices
- ROC curves
- Statistical comparisons between models

The code also generates visual comparisons and statistical analysis outputs to support experimental interpretation.

## Repository Structure

```text
cervical-cell-anomaly-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── paper/
│   └── manuscript.md
│
└── src/
    └── main.py
```

### Main Files

| File | Description |
|---|---|
| `README.md` | Project overview, methodology, and execution instructions |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Excludes temporary files, datasets, and model outputs |
| `paper/manuscript.md` | Research manuscript workspace |
| `src/main.py` | Main implementation and experimental pipeline |

## Installation and Execution

### Requirements

The project is implemented in Python and uses PyTorch and scientific computing libraries.

The required packages are listed in `requirements.txt`.

Main dependencies include:

- NumPy
- Pandas
- OpenCV
- Matplotlib
- Seaborn
- tqdm
- PyTorch
- Torchvision
- scikit-learn
- SciPy

A CUDA-enabled GPU is recommended for running the complete experimental pipeline.

### Running on Kaggle

The current implementation is configured to use Kaggle directory paths.

Follow these steps to run the project.

**Step 1 — Create a Kaggle Notebook**

Create a new notebook on Kaggle and select an environment with GPU acceleration.

**Step 2 — Add the Dataset**

Use the **Add Input** option in Kaggle to attach the SIPaKMeD dataset.

The current configuration expects the dataset at:

```text
/kaggle/input/datasets/marinaeplissiti/sipakmed
```

If Kaggle mounts the dataset at a different location, update the `dataset_root` value in the configuration section of `src/main.py`.

**Step 3 — Clone the Repository**

Run the following command in a Kaggle notebook cell:

```bash
!git clone https://github.com/haamedhasani/cervical-cell-anomaly-detection.git
```

**Step 4 — Enter the Project Directory**

```python
%cd /kaggle/working/cervical-cell-anomaly-detection
```

**Step 5 — Install Dependencies**

Kaggle usually provides PyTorch and several scientific libraries by default.

If the required dependencies are already available, this step can be skipped.

Otherwise, install the listed packages:

```bash
!pip install -r requirements.txt
```

Check PyTorch and Torchvision compatibility if installing or changing their versions.

**Step 6 — Run the Main Script**

```python
%run src/main.py
```

The script performs dataset loading, preprocessing, model training, anomaly detection, evaluation, statistical analysis, and visualization.

### Output Directory

The current configuration saves experimental outputs to:

```text
/kaggle/working/Unified_HybridVAEGAN
```

The generated outputs include evaluation results, comparison tables, statistical analysis files, plots, and visual comparisons.

The output directory can be changed through the `save_dir` parameter in `src/main.py`.

**Note:** The complete experimental pipeline may require substantial GPU memory and execution time. Actual runtime depends on the available hardware and dataset size.

## Reproducibility

Several settings are provided to support reproducible experiments:

- Fixed random seed (`42`)
- Group-aware data splitting
- Shared experimental configuration
- Consistent input image dimensions
- Defined optimization parameters
- Early stopping mechanisms

For reproducibility, users should also record the Python, PyTorch, Torchvision, CUDA, and hardware versions used in their experiments.

The current dependency file does not pin exact package versions. Therefore, identical results across different software and hardware environments are not guaranteed.

## Current Repository Status

This repository contains the main research implementation and its initial documentation.

The project is being prepared for scientific inspection, reproducibility, and further development.

Additional materials, including finalized experimental results, supplementary figures, and publication information, may be added as the research project progresses.

## Citation

If you use this implementation in academic research, please cite the repository and its associated research publication when available.

A permanent DOI and a formal citation entry will be added after the first archived release.

## License

A license has not yet been assigned to this repository.

Please contact the repository owner before redistributing or reusing the code until the licensing terms are formally specified.

The dataset is subject to its own terms of use and is not covered by any future software license assigned to this repository.

## Disclaimer

This repository is intended for academic research and experimental evaluation.

The implemented models are research tools and are not intended for clinical diagnosis, medical decision-making, or direct use in patient care.

---

**Repository:** [cervical-cell-anomaly-detection](https://github.com/haamedhasani/cervical-cell-anomaly-detection)
