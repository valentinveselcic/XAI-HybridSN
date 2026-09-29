# XAI-HybridSN: Explainable Hyperspectral Image Classification Using Hybrid 3D-2D CNN Architecture

This project implements a hybrid 3D and 2D convolutional neural network (HybridSN) for hyperspectral image (HSI) classification, employing explainable AI (XAI) methods (Grad-CAM and sensitivity analysis via spectral band perturbation) to interpret model decisions and quantify the impact of spectral wavelengths on classification accuracy.

## Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Repository Structure](#repository-structure)
* [Tech Stack](#tech-stack)
* [Prerequisites](#prerequisites)
* [Installation & Setup](#installation--setup)
* [Usage](#usage)
* [Results & Visualizations](#results--visualizations)
* [License](#license)

## Project Overview

Hyperspectral Imaging (HSI) gathers hundreds of narrow, contiguous spectral bands across the electromagnetic spectrum, enabling precise material identification based on unique spectral signatures. Due to the high dimensionality and non-linear interactions between spatial and spectral features, standard architectures often struggle to effectively capture joint representations. This project leverages the **HybridSN** model, which combines 3D convolutions (for spectral-spatial feature learning) and 2D convolutions (for abstract spatial representation learning).

To unpack the deep learning "black box", the pipeline integrates Explainable AI (XAI) techniques:

1. **Grad-CAM (Gradient-weighted Class Activation Mapping):** Visualizes the critical spatial regions driving class predictions by computing class gradients with respect to the feature maps of the final 2D convolutional layer (`conv2d`).
2. **Sensitivity Analysis (Spectral Band Perturbation / Zeroing):** Systematically eliminates (zeros out) individual raw spectral bands prior to PCA transformation to measure the drop in predicted class probabilities, quantifying the relative importance of each wavelength (430–860 nm).
3. **Misclassification Attribution:** Identifies specific spectral bands and wavelength ranges whose presence or absence causes the network to confuse classes with similar spectral signatures.

Experimental validation is conducted on the benchmark **Pavia University** dataset (acquired by the ROSIS sensor, containing 103 spectral bands across 9 urban land-cover classes).

## Features

* **Hybrid Spatial-Spectral Architecture:** Progressively extracts complementary spectral-spatial feature representations using three 3D-CNN layers followed by a 2D-CNN layer.
* **Dimensionality Reduction:** Applies Principal Component Analysis (PCA) with whitening to compress 103 raw spectral bands down to 15 principal components while preserving spatial structure.
* **3D Patch Extraction:** Segments HSI cubes into overlapping spatial-spectral patches of size $25 \times 25 \times 15$ with zero-padding to eliminate border artifacts.
* **Spatial Saliency Maps (Grad-CAM):** Computes class-specific activation heatmaps using linear layer modifiers to explain spatial localization.
* **Quantitative Band Importance Scoring:** Evaluates band contribution scores across all 103 channels for each class without retraining the model.
* **Misclassification Diagnosis:** Pinpoints spectral confusion factors between overlapping land-cover types (e.g., Gravel vs. Bare Soil vs. Self-Blocking Bricks).

## Repository Structure

```
├── HybridSN_V1.ipynb              # Main Jupyter Notebook containing the full pipeline
├── best-model.keras               # Serialized weights of the trained HybridSN model
├── pca_model.joblib               # Serialized Principal Component Analysis (PCA) transformer
├── classification_report.txt      # Classification report, accuracy metrics, and confusion matrix
├── band_importance_graphs/        # Visual plots of band importance scores per class
├── band_importance_scores/        # NumPy arrays (.npy) with numerical importance scores
├── GradCAM outputs/               # Saved Grad-CAM activation heatmaps per class
├── misclass_results/              # Error attribution analysis and confusion plots
├── .env.example                   # Environment configuration template
└── README.md                      # Project documentation
```

## Tech Stack

| Category | Technology / Library | Role |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Core programming language |
| **Deep Learning** | TensorFlow 2.x / Keras | Neural network design, training, and serialization |
| **XAI & Interpretability** | tf-keras-vis (Grad-CAM) | Class activation mapping and saliency visualization |
| **HSI Processing** | Spectral Python (SPy) | Hyperspectral data loading and false-color rendering |
| **Machine Learning** | Scikit-learn | PCA dimensionality reduction, train/test splitting, metrics |
| **Scientific Computing** | NumPy, SciPy | Tensor operations, array manipulations, and `.mat` parsing |
| **Data Visualization** | Matplotlib, Plotly | Spectral response graphs, training curves, and figures |
| **Interactive Environment**| Jupyter Notebook | Prototyping, running experiments, and evaluation |
| **Serialization** | Joblib | Caching spatial patches and saving the PCA model |

## Prerequisites

Ensure the following system and hardware requirements are met:

* **Operating System:** Linux (Ubuntu 20.04+) or Windows 10/11
* **Python:** Version 3.11 (managed via `pyenv` or `conda` recommended)
* **GPU Acceleration:** NVIDIA GPU supporting CUDA 11.8+ / 12.x and cuDNN (highly recommended for 3D convolutions)
* **RAM:** Minimum 16 GB system memory

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/valentinveselcic/OSIRV-Project.git
cd OSIRV-Project
```

### 2. Configure Virtual Environment

```bash
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install tensorflow keras tf-keras-vis spectral scikit-learn numpy scipy matplotlib plotly joblib tqdm notebook
```

### 4. Setup Environment Variables

Copy the template file and adjust configuration settings if necessary:

```bash
cp .env.example .env
```

Default content of `.env.example`:

```env
KERAS_BACKEND=tensorflow
CUDA_VISIBLE_DEVICES=0
DATASET_NAME=PU
WINDOW_SIZE=25
PCA_COMPONENTS=15
BATCH_SIZE=256
EPOCHS=100
```

### 5. Dataset Acquisition

The Pavia University dataset files (`PaviaU.mat` and `PaviaU_gt.mat`) can be downloaded automatically within the notebook or manually placed in the root directory from the [EHU Hyperspectral Remote Sensing Repository](http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes).

## Usage

All workflows are self-contained within `HybridSN_V1.ipynb`. Start the Jupyter server:

```bash
jupyter notebook HybridSN_V1.ipynb
```

The execution pipeline consists of:

1. **Data Loading & Preprocessing:** Loads the raw HSI tensor $(610 \times 340 \times 103)$, reduces spectral dimensions to 15 components using PCA, and extracts overlapping patches of size $25 \times 25$.
2. **Model Construction & Training:**
   * Three 3D convolutional layers:
     * `Conv3D(filters=8, kernel_size=(3, 3, 7), activation='relu')`
     * `Conv3D(filters=16, kernel_size=(3, 3, 5), activation='relu')`
     * `Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')`
   * Tensor reshaping to 2D followed by:
     * `Conv2D(filters=64, kernel_size=(3, 3), activation='relu')`
   * Fully connected classification head:
     * `Dense(256) -> Dropout(0.4) -> Dense(128) -> Dropout(0.4) -> Softmax(9)`
   * Optimized with Adam using an exponential decay schedule (`ExponentialDecay`, initial $lr = 0.001$).
3. **Model Evaluation:** Evaluates test metrics including Overall Accuracy (OA), Average Accuracy (AA), and Cohen's Kappa coefficient ($\kappa$).
4. **Grad-CAM Saliency Analysis:** Extracts gradients from the `conv2d` layer across all 9 classes and saves activation heatmaps to `GradCAM outputs/`.
5. **Channel Sensitivity Analysis:** Executes `class_band_importance_disk` to calculate sensitivity scores across all 103 bands via perturbation, outputting figures to `band_importance_graphs/` and data to `band_importance_scores/`.
6. **Error Analysis:** Runs `class_specific_misclassified_band_analysis` to identify spectral bands responsible for inter-class confusion.

## Results & Visualizations

### 1. Classification Performance

Evaluated on the test split (70% of total samples), the trained model achieves state-of-the-art results:

* **Overall Accuracy (OA):** $> 99.8\%$
* **Average Accuracy (AA):** $> 99.7\%$
* **Kappa Coefficient ($\kappa$):** $> 0.99$

### 2. Grad-CAM Spatial Localization

Grad-CAM heatmaps highlight model attention over physical ground features (e.g., roads, roofs, shadowed zones) while filtering out background noise:

| Class 1: Asphalt | Class 5: Painted Metal Sheets |
| :--- | :--- |
| <img src="GradCAM outputs/0_Asphalt_gradcam_outputs.jpg" width="350"/> | <img src="GradCAM outputs/4_Painted metal sheets_gradcam_outputs.jpg" width="350"/> |


*(Refer to `GradCAM outputs/` for all generated class maps)*

### 3. Spectral Band Importance Profiles

Ablation analysis through band zeroing identified physically consistent relationships:

* **Vegetation (Meadows, Trees):** High sensitivity in the red-edge region (700–750 nm) and near-infrared (NIR) plateau (~780 nm) driven by chlorophyll reflectance properties.
* **Man-Made Materials (Asphalt, Bitumen):** Distinct sensitivity across red wavelengths (620–750 nm) and specific absorption/reflection characteristics in the NIR spectrum (840 nm).
* **Spectral Confusion (Gravel vs. Bare Soil):** Importance profiles display high similarity in the 540–560 nm and 730–770 nm intervals, explaining minor classification boundary overlap.

Detailed charts and raw numerical matrices are stored under `band_importance_graphs/` and `band_importance_scores/`.

## License

This project is distributed under the terms of the **MIT License**.
