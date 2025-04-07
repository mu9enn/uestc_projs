# UESTC Projects

This repository, `uestc_projs`, serves as a collection of my coursework and competition projects completed during my studies at the University of Electronic Science and Technology of China (UESTC). It includes implementations of various algorithms, AI models, and games developed for educational purposes and competitive challenges.

## Project Structure

- **`ai_classworks/`**: Coursework related to Artificial Intelligence
  - `cnn_cifar10/`: CNN implementation for CIFAR-10 dataset
  - `connect_four/`: Connect Four game with AI agents
  - `fuzzy_wash/`: Fuzzy logic system for washing machine control
  - `gobang_game/`: Gobang (Five in a Row) game with AI
- **`garbage_class/`**: Project files related to garbage classification (e.g., zero-shot learning, fine-tuning)
- **`environment.yaml`**: Conda environment configuration for dependencies
- **`check_cuda.py`**: Utility script to check CUDA availability

## Getting Started

1. **Clone the Repository**  
   ```bash
   cd uestc_projs
   ```

2. **Set Up Environment**  
   Install dependencies using the provided `environment.yaml`:
   ```bash
   conda env create -f environment.yaml
   ```

3. **Run Projects**  
   Navigate to a specific project folder (e.g., `ai_classworks/cnn_cifar10/`) and follow the instructions in its `readme.md`.

## Notes
- Most projects are written in Python and may require libraries like PyTorch, NumPy, or Matplotlib.
- Some projects (e.g., `cnn_cifar10`) may need a GPU with CUDA support for optimal performance.

Feel free to explore, contribute, or adapt these projects for your own learning!
