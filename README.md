# ECG-Based Sex Classification Using Deep Learning

A machine learning project exploring whether biological sex can be predicted from 12-lead electrocardiogram (ECG) recordings using deep learning.

The project uses the **PTB-XL dataset** and PyTorch to develop, train, and evaluate neural network models for binary classification.

## Project Background

This project is a continuation of a research project I conducted during my MSc in Electrical Engineering at Aarhus University.

The original research investigated the use of deep learning to predict demographic characteristics, including sex and BMI, from ECG recordings.

With this new implementation, I aim to revisit the research using my current knowledge, improve the code structure and reproducibility, and explore different approaches to the classification task.

## Objectives

The main objectives of this project are:

- Build a reproducible pipeline for loading and preprocessing ECG recordings.
- Implement and evaluate deep learning models for sex classification.
- Compare model performance against simple baseline approaches.
- Explore how different architectures and training configurations affect performance.
- Document experimental results, limitations, and findings.

The initial focus is sex classification, with the possibility of extending the project to other prediction tasks later.

## Dataset

The project uses **PTB-XL**, a publicly available dataset of 12-lead ECG recordings.

- **Source:** [PTB-XL on PhysioNet](https://physionet.org/content/ptb-xl/)
- **Recordings:** Approximately 21,800 ECG recordings
- **Duration:** 10 seconds per recording
- **Sampling rates:** 100 Hz and 500 Hz
- **Prediction target:** Recorded sex (binary label)

The initial implementation uses the 100 Hz recordings, resulting in 12-channel signals with 1,000 samples per channel.

The dataset is not included in this repository and must be downloaded separately from PhysioNet.

## Technologies

- Python
- PyTorch
- NumPy and pandas
- WFDB
- scikit-learn
- Matplotlib
- pytest and Ruff

## Project Structure

```text
ecg-prediction/
├── configs/                # Experiment configurations
├── data/                   # Local datasets (not tracked by Git)
├── notebooks/              # Data exploration and experiments
├── src/ecg_prediction/
│   ├── dataset.py          # Dataset loading and splitting
│   ├── preprocessing.py    # ECG preprocessing
│   ├── model.py            # Neural network architectures
│   ├── train.py            # Model training
│   └── evaluate.py         # Model evaluation
├── tests/                  # Automated tests
├── pyproject.toml
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd ecg-prediction
```

### 2. Create a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -e ".[dev]"
```

For CPU-only PyTorch or a specific CUDA setup, refer to the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/).

### 4. Download PTB-XL

Download the dataset from [PhysioNet](https://physionet.org/content/ptb-xl/) and extract it into `data/ptb-xl/`.

The expected directory structure includes:

```text
data/ptb-xl/
├── ptbxl_database.csv
├── records100/
└── records500/
```

The `records500/` directory is optional for the initial implementation.

## Project Status

**Status: Early development**

The project is currently being rebuilt from scratch, starting with the dataset loading and preprocessing pipeline.

Planned development stages:

- [X] Implement PTB-XL dataset loading and validation
- [ ] Implement ECG preprocessing
- [ ] Establish baseline classification performance
- [ ] Implement a CNN-based classifier
- [ ] Reimplement the original CNN–LSTM architecture
- [ ] Develop a reproducible training pipeline
- [ ] Evaluate and compare model performance
- [ ] Document findings and limitations

## Limitations

This is an exploratory machine learning research project, not a clinical diagnostic tool.

The prediction target is the sex variable recorded in PTB-XL, which does not represent an individual's gender identity. Model results may reflect demographic imbalances, dataset characteristics, and other confounding factors.

## Acknowledgements

This project uses the PTB-XL dataset, originally published by Wagner et al. (2020):

*PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data, 7, 154.

Dataset DOI: [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).