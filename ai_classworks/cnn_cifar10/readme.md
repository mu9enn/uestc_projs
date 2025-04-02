# CIFAR-10 Evaluation Framework

This project provides a modular and extensible framework for evaluating classification performance on the CIFAR-10 dataset using different model architectures and training strategies, including:

- **ResNet-50 (RN50)** with and without ImageNet pretraining
- **CLIP** zero-shot classification
- **CLIP** with linear probing

## Project Structure

```
cifar10_eval/
├── data_preprocessing.py     # Load and preprocess CIFAR-10 dataset from OpenML
├── model.py                  # Define RN50 and CLIP models
├── train.py                  # Training loop and 10-fold cross-validation logic
├── main.py                   # Main entry to run all experiments
```

## Features

- 10-fold cross-validation
- Modular architecture (easy to plug in new models or evaluation methods)
- Image preprocessing pipeline for CLIP
- Zero-shot and linear probe evaluation modes

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- OpenML
- [CLIP](https://github.com/openai/CLIP) (install via `pip install git+https://github.com/openai/CLIP.git`)

## Usage

```bash
# Clone the repository and install requirements
git clone https://github.com/yourname/cifar10_eval.git
cd cifar10_eval
pip install -r requirements.txt

# Run all experiments (RN50, CLIP zero-shot, CLIP linear probe)
python main.py
```

## Model Options

The script evaluates the following settings:

- `RN50`: ResNet-50 with or without pretraining
- `CLIP_zeroshot`: CLIP using text-image similarity without training
- `CLIP_linear`: CLIP's frozen image encoder + trainable linear classifier

## Results

Each model's performance is printed per fold and averaged across 10 folds. Results can be easily exported or extended.

## Notes

- The dataset used is a subset of CIFAR-10 (`cifar_10_small`, ID: `40926`) loaded from OpenML.
- Images are stored as flat vectors of size 3072 (32×32×3), reshaped during preprocessing.

## License

MIT License

## Acknowledgements

- OpenAI CLIP
- PyTorch
- OpenML
- torchvision CIFAR-10 dataset


