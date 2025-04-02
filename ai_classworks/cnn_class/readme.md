# CIFAR-10 and HAM10000 Evaluation Framework

This project provides a modular and extensible framework for evaluating classification performance on the CIFAR-10 and HAM10000 datasets using various model architectures and training strategies, including:

- **ResNet-50 (RN50)** with and without ImageNet pretraining
- **CLIP** zero-shot classification
- **CLIP** with linear probing

The framework supports 10-fold cross-validation, advanced training techniques like mixed precision and cosine annealing learning rate scheduling, and is optimized for both datasets with minimal configuration changes.

## Project Structure

```
cifar_ham_eval/
├── data_preprocessing.py     # Load and preprocess CIFAR-10 and HAM10000 datasets
├── model.py                  # Define RN50 and CLIP models
├── train.py                  # Training loop and 10-fold cross-validation logic
├── main.py                   # Main entry to run all experiments
├── README.md                 # Project documentation (this file)
└── hmnist_28_28_L.csv        # HAM10000 dataset (not included, must be provided)
```

## Features

- **Dataset Support**: Handles CIFAR-10 (RGB, 32x32) and HAM10000 (grayscale, 28x28) datasets.
- **Model Flexibility**: Evaluates RN50 (with/without pretraining), CLIP zero-shot, and CLIP linear probing.
- **Training Enhancements**:
  - 10-fold cross-validation for robust performance evaluation.
  - Mixed precision training for efficiency on GPUs.
  - AdamW optimizer with cosine annealing learning rate scheduling.
  - Progress tracking with `tqdm`.
- **Modular Design**: Easily extendable to new models or datasets.

## Requirements

- Python 3.8+
- PyTorch (`pip install torch torchvision`)
- scikit-learn (`pip install scikit-learn`)
- OpenML (`pip install openml`) for CIFAR-10
- CLIP (`pip install git+https://github.com/openai/CLIP.git`)
- tqdm (`pip install tqdm`)
- pandas (`pip install pandas`) for HAM10000
- NumPy (`pip install numpy`)

Optional (for GPU support):
- CUDA-enabled PyTorch (see [PyTorch installation guide](https://pytorch.org/get-started/locally/))

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cifar_ham_eval
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision scikit-learn openml git+https://github.com/openai/CLIP.git tqdm pandas numpy
   ```

3. (Optional) For HAM10000, download `hmnist_28_28_L.csv` and place it in the project directory. This file is not included due to licensing and size constraints. Obtain it from the [HAM10000 dataset source](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000).

## Usage

Run the main script to execute all experiments:
```bash
python main.py
```

This will:
- Load CIFAR-10 from OpenML and HAM10000 from `hmnist_28_28_L.csv`.
- Perform 10-fold cross-validation for each dataset with:
  1. RN50 without pretraining
  2. RN50 with pretraining
  3. CLIP linear probing
  4. CLIP zero-shot classification
- Output mean accuracy for each experiment.

### Customizing Experiments
- **Change Dataset**: Modify `dataset_name` in `main.py` calls to `"cifar"` or `"ham10000"`.
- **Adjust Epochs**: Set `num_epochs` in `perform_10_fold_cv` calls.
- **HAM10000 Labels**: Update `ham_labels` in `main.py` with actual class names for zero-shot classification (currently placeholders).

## Technical Details

- **Data Preprocessing**:
  - CIFAR-10: Loaded via OpenML, 3072 features (3x32x32 RGB), 10 classes.
  - HAM10000: Loaded from CSV, 784 features (1x28x28 grayscale), variable classes (determined dynamically).
- **Training**:
  - Optimizer: AdamW with initial LR=1e-4.
  - Scheduler: Cosine annealing over the total number of epochs.
  - Mixed Precision: Uses `torch.cuda.amp` for faster training on GPUs.
- **Models**:
  - RN50: Adjusted final layer to match the number of classes.
  - CLIP Linear Probe: Freezes CLIP RN50 backbone, trains a linear classifier on 1024-dimensional features.
  - CLIP Zero-shot: Uses text prompts for classification without training.

## Notes

- **HAM10000 Class Labels**: The zero-shot experiment uses placeholder labels (`class_0`, `class_1`, etc.). Replace these with actual class names from the HAM10000 dataset for meaningful zero-shot results.
- **Performance**: Training time depends on hardware (CPU/GPU) and batch size (default: 1024). Adjust `batch_size` in `main.py` if memory issues occur.
- **Limitations**: 
  - Assumes `hmnist_28_28_L.csv` is in the project root. Update `file_path` in `main.py` if located elsewhere.
  - CLIP zero-shot requires meaningful text prompts, which are dataset-specific.

## Contributing

Feel free to submit issues or pull requests to enhance the framework, such as adding new models, datasets, or training strategies.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details (not included in this repo yet).