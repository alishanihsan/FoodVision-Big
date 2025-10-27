# FoodVision-Big

A collection of Jupyter Notebooks for exploring, training, and evaluating computer vision models on large-scale food image datasets. This repository provides end-to-end notebooks covering data exploration, preprocessing, model training (transfer learning and from-scratch), evaluation, and deployment-ready export steps. The notebooks are intended to be readable, reproducible, and easy to adapt to your own datasets and experiments.

> Note: This repository contains Jupyter Notebook files (.ipynb). The exact dataset(s) used by the notebooks are not bundled here — you should supply or download the dataset(s) and point the notebooks to the correct local paths or cloud storage.

Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Quick Start (local)](#quick-start-local)
  - [Quick Start (Docker, optional)](#quick-start-docker-optional)
- [Notebooks (High-level descriptions)](#notebooks-high-level-descriptions)
- [Working with Data](#working-with-data)
  - [Dataset suggestions](#dataset-suggestions)
  - [Expected layout](#expected-layout)
- [Model Training & Evaluation](#model-training--evaluation)
  - [Tips for reproducible experiments](#tips-for-reproducible-experiments)
- [Exporting & Deployment](#exporting--deployment)
- [Best Practices and Performance Tips](#best-practices-and-performance-tips)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

FoodVision-Big is a notebook-driven project focused on building accurate and robust food classification models on large image datasets. It contains a set of notebooks that guide you from data inspection to model export. The workflows are intended to be framework-agnostic where possible and include examples using common libraries (TensorFlow / Keras and PyTorch). The primary goals are:

- Provide clear, reproducible notebook examples for large-scale food image classification.
- Demonstrate practical preprocessing and augmentation pipelines for food images.
- Show how to use transfer learning effectively and compare with training from scratch.
- Provide evaluation scripts and guidance for deployment/export.

---

## Repository Structure

The repository is organized around Jupyter Notebooks and supporting assets. Typical structure:

- notebooks/ or root directory
  - EDA.ipynb — Exploratory Data Analysis and dataset stats
  - preprocessing.ipynb — Data cleaning and image transforms (augmentation)
  - training_tf.ipynb — Training example using TensorFlow / Keras (transfer learning)
  - training_pt.ipynb — Training example using PyTorch (transfer learning)
  - evaluation.ipynb — Evaluation, confusion matrices, metrics and analysis
  - export_model.ipynb — Exporting models to SavedModel / TorchScript / ONNX
  - inference_examples.ipynb — Example inference code for local / REST / batch
- assets/
  - README or small helper files (if present)
- docs/
  - Additional documentation notebooks or markdown (if present)
- requirements.txt (optional) — Python dependencies (create if not present)

Note: Notebooks may be named differently in the repo — check the root for the existing .ipynb files.

---

## Key Features

- Notebook-first workflows for quick iteration and education.
- Transfer learning examples using state-of-the-art pretrained backbones.
- Practical data augmentation and preprocessing recipes suited for food images (color preservation, crop/scale strategies).
- Model evaluation and visualization (confusion matrix, top-k accuracy, per-class metrics).
- Export and inference examples to run models in production or on edge devices.

---

## Getting Started

These steps assume you have Python and a GPU (optional but recommended for training) available.

### Requirements

- Python 3.8+ (3.9/3.10 recommended)
- Jupyter or JupyterLab
- Common ML libraries:
  - numpy, pandas, matplotlib, seaborn, scikit-learn
  - pillow (PIL)
  - one or both deep learning frameworks:
    - TensorFlow 2.x (tensorflow and tensorflow-io if needed)
    - PyTorch (torch, torchvision)
  - albumentations (recommended for advanced augmentations)
  - tqdm (progress bars)
  - opencv-python (cv2) — optional but useful for image processing

If a requirements.txt is present, install with:
```bash
python -m pip install -r requirements.txt
```

If not, use a minimal recommended pip install:
```bash
python -m pip install jupyterlab numpy pandas matplotlib seaborn scikit-learn pillow tqdm opencv-python
# and at least one of:
python -m pip install "tensorflow>=2.6"     # for TensorFlow-based notebooks
# or
python -m pip install torch torchvision     # for PyTorch-based notebooks
```

GPU users: install CUDA-enabled versions of the frameworks according to official docs (e.g., pip/conda instructions for tensorflow or torch that match your CUDA).

### Quick Start (local)

1. Clone the repository:
```bash
git clone https://github.com/alishanihsan/FoodVision-Big.git
cd FoodVision-Big
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install dependencies:
```bash
python -m pip install -r requirements.txt
```
(If the file doesn't exist: see the Requirements section above.)

4. Launch JupyterLab:
```bash
jupyter lab
```
Open the notebooks and follow step-by-step instructions inside each notebook.

### Quick Start (Docker, optional)

Create a Dockerfile or use an existing ML image from NVIDIA or the official TensorFlow/PyTorch images. Example (high-level):

```bash
docker run --gpus all -it -p 8888:8888 -v "$(pwd)":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:xx.xx-py3 jupyter lab --ip=0.0.0.0 --allow-root --no-browser
```

Adjust the image and commands as needed. Using Docker ensures consistent environments across machines.

---

## Notebooks (High-level descriptions)

- EDA.ipynb
  - Dataset overview, class distribution, sample visualization, dataset imbalance analysis.
- preprocessing.ipynb
  - Image resizing, normalization, augmentation pipelines (flip, crop, color jitter), and dataset builders.
- training_tf.ipynb
  - Transfer learning example using TensorFlow/Keras (fine-tuning pretrained CNNs, callbacks, learning rate schedules).
- training_pt.ipynb
  - Transfer learning example using PyTorch (datasets, dataloaders, training loop, checkpointing).
- evaluation.ipynb
  - Computing accuracy, precision/recall/F1, confusion matrices, class-wise analysis, and error visualization.
- export_model.ipynb
  - Converting and saving models to production-friendly formats (SavedModel, TorchScript, ONNX).
- inference_examples.ipynb
  - Running inference locally, batched inference, and example REST-serving snippets.

If your repository has different notebook filenames, open them and follow the internal documentation cells.

---

## Working with Data

### Dataset suggestions

Common public large-scale food image datasets you can experiment with:
- Food-101 — 101 food categories (good for benchmarking)
- Your own dataset collected from a source with labels
- Datasets from Kaggle (if applicable)

Make sure you respect dataset licenses and privacy rules.

### Expected layout

Many notebooks expect a dataset layout similar to:

- dataset/
  - train/
    - class_001/
      - img001.jpg
      - img002.jpg
      - ...
    - class_002/
      - ...
  - val/
    - class_001/
    - class_002/
  - test/
    - class_001/
    - ...

If your dataset is in CSV format with image paths and labels, the notebooks generally include an example to convert or generate the expected folder-structured format.

---

## Model Training & Evaluation

- Use transfer learning (pretrained backbones such as EfficientNet, ResNet, MobileNet, Vision Transformers) when dataset size is moderate. Fine-tune top layers first, then unfreeze and fine-tune deeper layers if necessary.
- Track experiments with a logging tool (TensorBoard, Weights & Biases, MLflow) — there are snippets in the notebooks for TensorBoard.
- Save checkpoints and use early stopping to prevent overfitting.
- Evaluate with top-1 and top-5 accuracy for multi-class classification; use per-class metrics for imbalanced datasets.

### Tips for reproducible experiments

- Set random seeds for numpy, python's random, and the framework (torch.manual_seed / tf.random.set_seed).
- Log package versions (e.g., pip freeze > requirements_freeze.txt).
- Save the exact notebook or convert to a script with the environment details for reproducibility.

---

## Exporting & Deployment

- For TensorFlow: Save model as a SavedModel and serve via TensorFlow Serving or convert to TensorFlow Lite for edge devices.
- For PyTorch: Export to TorchScript (script or trace) or ONNX for interoperability.
- Example commands and notebook cells show how to perform these exports and run simple inference scripts.

---

## Best Practices and Performance Tips

- Use mixed precision training (float16) to speed up training on compatible GPUs.
- Use image caching pipelines and optimized data loaders (prefetching, num_workers) to keep GPUs saturated.
- Start with a small input size and increase only if accuracy suffers significantly.
- Use robust augmentations to improve generalization, but be cautious with augmentations that change the food semantics (e.g., extreme color shifts).

---

## Contributing

Contributions are welcome. Suggested ways to contribute:

- Add or improve notebooks (clarity, reproducibility, additional frameworks).
- Provide a requirements.txt or environment.yml describing tested dependencies.
- Add dataset download & preprocessing utilities (or links to dataset sources).
- Add CI checks (e.g., ensuring notebooks run or converting them to .py for unit tests).
- Open issues or pull requests with clear descriptions and reproducible steps.

When contributing, please follow standard GitHub etiquette:
- Fork the repository
- Create a feature branch
- Open a PR with a clear description of changes
- Include tests or runnable examples where applicable

---

## License

Specify the repository license here (e.g., MIT, Apache-2.0). If a LICENSE file exists in this repository, follow that license. If no license is present, the default is "All rights reserved" — add a license file to make reuse explicit.

Example: add a LICENSE file with the MIT license if you want permissive reuse.

---

## Contact

Author: alishanihsan
Repository: https://github.com/alishanihsan/FoodVision-Big

If you have questions, issues, or feature requests, please open an issue in the repository.

---

## Acknowledgements

- Public datasets and pretrained models provided by the community (e.g., Food-101, ImageNet models).
- Open-source libraries: TensorFlow, PyTorch, torchvision, albumentations, scikit-learn, and others used throughout the notebooks.

---

If you'd like, I can:
- Generate a requirements.txt or conda environment file tailored to the notebooks found in this repo.
- Add example dataset download/organizer notebook.
- Convert one of the notebooks into a runnable Python script and create a small CI workflow to validate it.

Tell me which you'd prefer and I can prepare it.
